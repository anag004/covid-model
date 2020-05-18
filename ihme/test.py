import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from curvefit.core.model import CurveModel
from curvefit.core.functions import ln_gaussian_cdf, gaussian_cdf
from scipy.stats import norm
from sklearn.linear_model import LinearRegression

# Fetches the data from the COVID19India website
class DataFetcher:
    def __init__(self, url="https://api.covid19india.org/data.json"):
        self.url = url
        self.json_data = None
        self.cases_time_series = None
        
    def fetch(self):
        r = requests.get(url=self.url)
        self.json_data = r.json()
        
        # NOTE : start date of this data is 30 Jan
        
        # Get the fields
        fields = list(self.json_data['cases_time_series'][0].keys())
        self.cases_time_series = {}
        
        for field in fields:
            if field == 'date':
                self.cases_time_series[field] = [x[field] for x in self.json_data['cases_time_series']]
            else:
                self.cases_time_series[field] = np.array([float(x[field]) for x in self.json_data['cases_time_series']])

class GaussianCDF:

    """Fit a single Gaussian Atom to cumulative daily deaths"""
    
    def __init__(self):
        self.model = None
    
    def fit(self, daily_deaths, social_distance=None):
        daily_deaths = np.array(daily_deaths)
        n_data = daily_deaths.shape[0]
        
        # Prepare the data frame
        df = pd.DataFrame()
        df['death_rate'] = np.cumsum(daily_deaths)
        df['time'] = np.arange(df['death_rate'].shape[0])
        df['ln_death_rate'] = np.log(df['death_rate'] + 1) # Add 1 to pad in case the #deaths are zero
        df['group'] = ['all'] * n_data
        df['cov_one'] = [1.0] * n_data 
        
        
        if social_distance is not None:
            df['social_distance'] = social_distance 
            col_covs = [['cov_one'], ['cov_one', 'social_distance'], ['cov_one']] 
            num_fe = 4
            fe_init = [-3, 100, 0, 10]
            # col_covs = [['cov_one', 'social_distance'], ['cov_one', 'social_distance'], ['cov_one', 'social_distance']] 
            # num_fe = 6
            # fe_init = [-3, 0, 100, 0, 10, 0]
        else:
            col_covs = [['cov_one'], ['cov_one'], ['cov_one']] 
            num_fe = 3
            fe_init = [-3, 100, 1]

        # Set up the CurveModel
        self.model = CurveModel(
            df=df,
            col_t='time',
            col_obs='ln_death_rate',
            col_group='group',
            col_covs=col_covs,
            param_names=['alpha', 'beta', 'p'],
            link_fun=[lambda x: np.exp(x), lambda x: x, lambda x: np.exp(x)],
            var_link_fun=[lambda x: x] * num_fe,
            fun=ln_gaussian_cdf
        )
        
        
        # Fit the model to estimate parameters
        self.model.fit_params(
            fe_init=fe_init,
            options={
                'ftol': 1e-14,
                'maxiter': 500
            },
            re_bounds= [[0, 0]] * num_fe    # No random effects
        )
        
    
    def predict(self, t):
        """Get predictions for values in t"""
        
        return self.model.predict(t=t, group_name='all')
    
    def get_params(self):
        return np.squeeze(self.model.params)


# Get dataframe for the entire country from 25th Feb
# Although the death data starts from 14 March, we use mobilities from 25th Feb
# This is because the mean time to death is 18 days and so we shift the mobility data by 18 days forward
df = pd.read_csv("Global_Mobility_Report.csv", low_memory=False)
df = df[df['country_region'] == 'India']
df = df[pd.isnull(df['sub_region_1'])]
df = df[df['date'] >= '2020-02-25']
df.keys()

fetcher = DataFetcher()
print("Fetching data...")
fetcher.fetch()
print("Done")

# Exclude these days many days from the starting of the cases time series
offset = 43
death_data = fetcher.cases_time_series['dailydeceased'][offset:]
locations = ['retail_and_recreation',
             'grocery_and_pharmacy',
             'parks',
             'transit_stations',
             'workplaces',
             'residential']

# Make the lengths of the data equal
if len(df) > len(death_data):
    df = df[:-(len(df) - len(death_data))]
else:
    for i in range(len(death_data) - len(df)):
        df = df.append(df.iloc[-1])
        
social_distance = np.zeros(len(df))
for loc in locations:
    social_distance += np.array(df['{}_percent_change_from_baseline'.format(loc)])

model_india = GaussianCDF()
model_india.fit(death_data, social_distance)

time_values = np.arange(len(df))
pred = model_india.predict(time_values)

print("Extended params {}".format(model_india.model.expanded_params))

preds = np.exp(ln_gaussian_cdf(time_values, model_india.model.expanded_params))
print("Predictions {}:".format(preds))

plt.plot(time_values, preds, label="Predicted deaths")
plt.plot(time_values, np.cumsum(death_data), '.', label="Actual deaths")
plt.title("India | Single Gaussian Atom")
plt.xlabel("No. of days (since 14 March)")
plt.ylabel("Cumulative no. of Deaths")
# plt.legend()
plt.show()
