#imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from scipy.integrate import solve_ivp
import datetime
from copy import deepcopy
from scipy.stats import norm
from scipy.optimize import differential_evolution
import pdb
from IPython.core.display import HTML
from math import ceil, floor
from itertools import chain
from scipy.signal import savgol_filter
from scipy.special import expit
from scipy.integrate import solve_ivp
from itertools import product
import tqdm
import os
from pathlib import Path

class DataFetcher:
    """

    Fetch data for the entire country from covid19india. 
    
    Example usage:

    ```
    fetcher = DataFetcher() 
    fetcher.fetch()             # This command retrieves and processes data 
    fetcher.cases_time_series   # Cumulative confirmed cases are now stored in this field
    ```

    """

    def __init__(self, url="https://api.covid19india.org/data.json"):
        self.url = url
        self.json_data = None
        self.cases_time_series = None
        
    def fetch(self):
        r = requests.get(url=self.url)
        self.json_data = r.json()
        
        # Get the fields
        fields = list(self.json_data['cases_time_series'][0].keys())
        self.cases_time_series = {}
        
        for field in fields:
            if field == 'date':
                self.cases_time_series[field] = [x[field] for x in self.json_data['cases_time_series']]
            else:
                self.cases_time_series[field] = np.array([float(x[field]) for x in self.json_data['cases_time_series']])
        
    def train_data(self, threshold):
        self.fetch()
        index = np.where(self.cases_time_series["totaldeceased"] > threshold)[0][0]
        startdate = self.cases_time_series["date"][index]
        return self.cases_time_series["totaldeceased"][index:], startdate

class DataFetcherState:
    """
        Fetch state-wise COVID19 data from the covid19india website

        Example usage:

        ```
        fetcher = DataFetcher() 
        fetcher.fetch()             # This command retrieves and processes data 
        ```
        
        The `fetch` method retrives and processes data and stores it in the `data` dictionary. E

        Example usage:

        ```
        fetcher.data['mh']['deceased']      # Cumulative number of deceased people in maharashtra
        fetcher.data['dl']['recovered']     # Cumulative number of recovered people in delhi
        fetcher.data['gj']['confirmed']     # Cumulative number of confirmed cases in Gujarat
        ```
    """
    
    def __init__(self):
        self.data = None
        self.raw_data = None
    
    def fetch(self):
        # Fetch the raw data
        r = requests.get(url="https://api.covid19india.org/states_daily.json")
        self.raw_data = r.json()
        self.data = {}
        
        # Iterate over the days and record the data
        for entry in self.raw_data['states_daily']:
            status = entry['status'].lower()
            for state in entry:
                if state == "date" or state == "status":
                    continue
                    
                if state not in self.data:
                    # Initialize this state
                    self.data[state] = {
                        'deceased' : [],
                        'recovered': [],
                        'confirmed': []
                    }
                
                # Append the data
                self.data[state][status].append(entry[state])
                
    def start_date(self):
        return self.raw_data['states_daily'][0]['date']

def smoothen(X, std):
    """
    Convolve `X` with a Gaussian pdf with standard deviation `std` and return the smooth data 
    """

    result = np.zeros_like(X)
    
    for i, _ in enumerate(X):
        norm_factor = 0
        for j, _X in enumerate(X):
            val = norm.pdf(j, loc=i, scale=std)
            result[i] += _X * val
            norm_factor += val
        result[i] /= norm_factor
    
    return result

def rmse_loss(x, y):
    """
    Return the (scalar) root mean squared loss between `x` and `y`. 
    Note that `x` and `y` must be arrays of the same shape.
    """

    return np.sqrt(np.mean(np.square(x - y)))

def run_model(*params, return_preds=False, integrate=False):
    """
    Given a model, loss function and daily death data, along with the parameters, run the model on these parameters.

    Arguments
    --------------
    params        : A list of parameters. The last three must be the model function, loss function and daily death data
    return_preds  : If False the function only returns the loss otherwise it returns both the loss and predictions
    integrate     : Doesn't do anything, present for compatibility reasons

    Returns
    --------------
    If return_preds = False, returns a single scalar value - the loss between the model predictions and actual data. 
    If return_preds = True, returns a tuple of the loss and predictions made by the model
    """

    model, loss_fun, daily_deaths = params[-3:]
    
    offset = int(params[0][0])
    t_max = daily_deaths[offset:].shape[0] - 1
    preds = model(t_max, *params[0], integrate=integrate)

    loss = loss_fun(preds, daily_deaths[offset:])
    
    if not return_preds:
        return loss
    else:
        return loss, preds

def fit_model(*model_args, bounds=None, verbose=True):
    """
    Return best fit parameters for a model using the differential evolution algorithm (slow)
    """

    def print_progress(xk, convergence=0):                
        if verbose:
            np.set_printoptions(
                formatter={
                    'float': lambda x: "{:.1e}".format(x)
                }
            )
            
            if xk.shape[0] > 6:
                print("x0: {} conv: {:.3e}".format(xk[:6], convergence), end="\r")
            else:
                print("x0: {} conv: {:.3e}".format(xk, convergence), end="\r")


    result = differential_evolution(run_model,
                                    args=model_args,
                                    bounds=bounds,
                                    callback=print_progress)

    print(result.message)
    return result.x

def get_model_stats(model, loss_fun, data, breakpoints, plot_title, bounds, param_cols, huge=False):
    """
    Given a model, loss function, data and breakpoints this function iterates through the breakpoints and plots the best fit curves for different parts of the data. It also prints a table of inferred parameters. This function uses the differential evolution algorithm and is slow. 
    """


    df = pd.DataFrame()
    df['Breakpoint'] = []
    df['Loss']       = []
    
    
    for col in param_cols:
        df[col] = []
    
    time_values = np.arange(data.shape[0])
    
    if huge:
        col_num = 2
        row_num = ceil(len(breakpoints) / col_num)
        fig, axs = plt.subplots(row_num, col_num, figsize=(25, 10 * row_num), sharex=True, sharey=True)
    else:
        col_num = 3
        row_num = ceil(len(breakpoints) / col_num)
        fig, axs = plt.subplots(row_num, col_num, figsize=(15, 4 * row_num), sharex=True, sharey=True)
    
    for b, ax in tqdm.tqdm(zip(breakpoints, axs.flat)):
        params = fit_model(model, loss_fun, data[:b], bounds=bounds, verbose=True)
        offset = int(params[0])
        
        loss, preds = run_model(params, model, loss_fun, data, return_preds=True)
  
        ax.plot(time_values[offset:], preds, label="Projected deaths")
        ax.scatter(time_values[:b], data[:b], s=2, c='green', label="Used points")
        ax.scatter(time_values[b:], data[b:], s=2, c='red', label="Unused points")
        ax.axvspan(b, time_values[-1], facecolor='r', alpha=0.2)
        ax.set_title("{} days".format(b))
        ax.set_xlabel("Days")
        ax.set_ylabel("Deaths")
        ax.legend()
        

        df_dict = {
            "Breakpoint": [b],
            "Loss"      : [loss],
        }

        for i, col in enumerate(param_cols):
            df_dict[col] = [params[i]]

        df = df.append(pd.DataFrame(df_dict))
    
    fig.suptitle(plot_title, fontsize=15)
    fig.tight_layout(pad=2.0)
    return df

def SIRD_fsm(t, y, R_t, T_inf, gamma_x, *R_args):
    """
        This is a finite state machine which given the S, I, X value at time t, computes these values at time t + 1

        Arguments
        ----------------
        t       : The current time in days
        y       : List of size 3 containing the current S, I, X values
        R_t     : Either a scalar or a function of the form R_t(t, *params) where t is the current time 
        T_inf   : The average time a person with COVID19 is infectious
        gamma_x : The number of people who die from COVID19 in a day as a fraction of the number of infected people
        R_args  : List of arguments to be passed onto the function calculating R

        Returns 
        ----------------
        A list of size 3 containing the values S, I, X at time t + 1
    """

    if callable(R_t):
        R = R_t(t, *R_args)
    else:
        R = R_t
    
    S, I, X = y
    
    S_out = S - (R * I * S / T_inf)
    I_out = I + (R * I * S / T_inf) - I / T_inf
    X_out = X + gamma_x * I

    return [S_out, I_out, X_out]

def sird_sd(t_max, offset, pop,  I_init, T_inf, gamma_x, R_max, R_min, sd_offset, padded_sd, integrate=False):
    """ 
    Returns predictions from the SIRD model. Assumes that the social distancing data is available in a global array `padded_sd`

    Arguments
    ------------
    t_max     : The time until which predictions have to be made
    offset    : At what index into the data should the model be started
    pop       : The total population of the region
    I_init    : The initial number of infections as a fraction of the total population
    T_inf     : The average time for which a person stays infectious
    gamma_X   : The number of people who die from COVID19 in a day as a fraction of the number of infected people
    R_max     : The value of R before the lockdown
    R_min     : The value of R after the lockdown is put into place
    sd_offset : The amount (in days) the social distancing data lags behind the death count
    padded_sd : The social distancing data, padded appropriately
    integrate : Default value is False. If True, the model uses Runge-Kutta methods to numerically solve DEs (slow)

    Returns 
    ------------
    An array of size (t_max + 1) with the predictions from the model
    """

    S = 1
    I = I_init
    X = 0
    
    if not integrate:
        result = np.zeros(t_max + 1, dtype=float)
        result[0] = 0
    #   This computes values using a FSM
        for t in range(1, t_max + 1):
            S, I, X = SIRD_fsm(t, [S, I, X], R_sd_v2, T_inf, gamma_x, R_max, R_min, sd_offset, padded_sd, offset)
            result[t] = X 
    else:
        # This computes values using a Runge-Kutta method of order 5
        integral = solve_ivp(fun=SIRD_fsm, 
                             t_span=(0, t_max + 1),
                             t_eval=np.arange(0, t_max + 1),
                             y0=[S, I, X],
                             args=(R_sd_v2, T_inf, gamma_x, R_max, R_min, sd_offset, padded_sd, offset))
        result = integral.y[2]
    
    result[1:] = np.diff(result)
    return result * pop

def get_model_stats_v2(model, loss_fun, data, breakpoints,  
                       fixed_params, var_param_vals, param_order, loss_factor, future_preds=0, huge=False, plot_title=None, filename=None, exclude_params=None):
    
    """ 
    Newer version of get_model_stats, includes uncertainty analysis. Fits a given model, plots and saves projections.

    Arguments
    -------------
    model           : Callable function of the form f(t_max, *params) where params are the model parameters. Function should return predictions in an array.
    loss_fun        : Function of form f(x, y) which returns the loss between vectors x, y
    data            : An array containing the daily death counts
    breakpoints     : An array of time values at which the data should be split into train/test sets. A plot is produced for each breakpoint
    plot_title      : The title of the set of plots. If left empty, nothing is plotted
    filename        : The path where the predictions should be stored in JSON format. If unspecified, predictions are not stored.
    fixed_params    : A dictionary with string keys and float values mapping parameter names to their fixed values. 
    var_param_vals  : A dictionary with string keys and list values mapping variable parameter names to their possible values.  
    param_order     : A list of string names of parameters specifying the order in which they are passed into the model function
    loss_factor     : Controls the size of the uncertainty interval. Corresponds to `\alpha` in the whitepaper
    future_preds    : The number of days into the future the model should predict deaths
    huge            : If this is True large plots are produced
    exclude_params  : List of params which should not be saved

    Returns
    -------------
    A pandas DataFrame with the inferred model parameters and projections. 
    """


    df = pd.DataFrame()
    df['Breakpoint'] = []
    df['Train Loss'] = []
    
    for col in list(fixed_params.keys()) + list(var_param_vals.keys()):
        df[col] = []
    
    time_values = np.arange(data.shape[0])
    
    col_num = 3
    row_num = ceil(len(breakpoints) / col_num)
    
    if plot_title is not None:
        if not huge:
            fig, axs = plt.subplots(row_num, col_num, figsize=(15, 4 * row_num), sharex=True, sharey=True)
        else:
            col_num = 2
            row_num = ceil(len(breakpoints) / col_num)
            fig, axs = plt.subplots(ceil(len(breakpoints) / 2), 2, figsize=(15, 8 * row_num), sharex=True, sharey=True)

        loop_iterator = zip(breakpoints, axs.flat)
    else:
        loop_iterator = zip(breakpoints, range(len(breakpoints)))
        
    var_param_list = list(var_param_vals.keys())
    
    for b, ax in tqdm.tqdm(loop_iterator):
        min_loss = np.inf
        best_preds = None
        best_offset = None
        best_params = None
        all_preds = []
        
        for vals in product(*var_param_vals.values()):
            # Run the model, get loss and preds
            t_max = data[:b].shape[0] - 1
            
            params = []
            for x in param_order:
                if x in fixed_params:
                    params.append(fixed_params[x])
                elif x in var_param_vals:
                    params.append(vals[var_param_list.index(x)])
            
            loss, _ = run_model(params, model, loss_fun, data[:b], return_preds=True)
            _, preds = run_model(params, model, loss_fun, np.concatenate((data, np.zeros(future_preds))), return_preds=True)
            
            if "offset" in fixed_params:
                offset = fixed_params["offset"]
            elif "offset" in var_param_vals:
                offset = vals[var_param_list.index("offset")]
            
            if loss < min_loss:
                min_loss = loss
                best_preds = preds
                best_offset = offset
                best_params = params
            
            all_preds.append((loss, preds, offset))


        # Get the uncertainty interval 
        min_preds = np.ones_like(best_preds) * np.inf
        max_preds = np.ones_like(best_preds) * -np.inf

        for loss, preds, offset in all_preds:
            if min_loss + loss_factor / (b - offset) >= loss:
                min_preds = np.minimum(min_preds, preds)
                max_preds = np.maximum(max_preds, preds)

        avg_preds = 0.5 * (min_preds + max_preds)
  
        if plot_title is not None:
            ax.scatter(time_values[:b], data[:b], s=2, c='green', label="Used points")
            ax.scatter(time_values[b:], data[b:], s=2, c='red', label="Unused points")
            ax.axvspan(b, time_values[-1], facecolor='r', alpha=0.2)

            ax.plot(best_offset + np.arange(best_preds.shape[0]), avg_preds, label="Projected deaths")
            ax.fill_between(best_offset + np.arange(best_preds.shape[0]), min_preds, max_preds, alpha=0.2)
            ax.set_title("{} days".format(b))
            ax.set_xlabel("Days")
            ax.set_ylabel("Deaths")
            ax.legend()
        
        if filename is not None:
            df_dict = {
                "Breakpoint"  : [b],
                "Train Loss"  : [min_loss],
                "preds"       : [avg_preds],
                "minpreds"    : [min_preds],
                "maxpreds"    : [max_preds]
            }
        else:
            df_dict = {
                "Breakpoint"  : [b],
                "Train Loss"  : [min_loss],
                "preds"       : [avg_preds],
            }

        for i, col in enumerate(param_order):
            if exclude_params is None or col not in exclude_params: 
                df_dict[col] = [best_params[i]]

        df = df.append(pd.DataFrame(df_dict))

    if plot_title is not None:
        fig.tight_layout(pad=2.0)
        fig.suptitle(plot_title)
    
    if filename is not None:
        directory = os.path.dirname(filename)
        Path(directory).mkdir(exist_ok=True, parents=True)
        df.reset_index(inplace=True)
        json_data = df.to_json()

        with open(filename, "w") as f:
            f.write(json_data)

    return df

def R_sd_v2(t, R_max, R_min, sd_offset, sd_metric, offset):    
    """
    Returns R as a function of time

    Arguments
    ------------
    t         : The time in days
    R_max     : The value of R before the lockdown
    R_min     : The value of R when the lockdown is strictest
    sd_offset : The number of days the social distancing data lags the death data
    sd_metric : An array containing the social distancing data
    offset    : The offset into the death data from where the model starts

    Returns 
    ------------
    A float which is the R value at time t
    """


    sd_offset = int(sd_offset)
    offset = int(offset)
    
    sd_max = np.max(sd_metric)
    sd_min = np.min(sd_metric)
    
    if t + sd_offset + offset >= sd_metric.shape[0]:
        sd = sd_metric[-1]
    else:
        shifted_t = t + sd_offset + offset
        low_sd = sd_metric[floor(shifted_t)]
        high_sd = sd_metric[ceil(shifted_t)]
        sd = low_sd + (shifted_t - floor(shifted_t)) * (high_sd - low_sd)
    
    return R_max - (R_max - R_min) * (sd_max - sd) /  (sd_max - sd_min)