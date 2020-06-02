# A transparent SIRD model for COVID19 death projections in India

We are a group of 2 students from IIT Delhi who, observing the rapidly worsening COVID19 crisis in India, recognized the urgent need to come up with effective mitigation strategies. To stress this point, we developed an SIRD compartmental model to predict the spread of the disease in India. We believe that any modelling strategy is only as good as its assumptions. Therefore any honest attempt at modelling must strive to be as transparent as possible so that the community can easily replicate and examine the projections and assumptions involved. 

## Model details

An SIRD model is a type of [compartmental model](https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology). Compartmental models partition the population into group or _compartments_. An SIRD is a simple of compartmental model which has just four compartments. We encourage you to read about our model in detail in this [whitepaper](https://github.com/anag004/covid-model/blob/master/ihme/whitepaper.pdf) we have released. 

## Data sources

We have three sources of data

- [The John Hopkins CSSE COVID19 repo](https://github.com/CSSEGISandData/COVID-19) - We use this data source for daily death counts in countries and regions outside India.
- [covid19india tracker project](https://www.covid19india.org) - This is a volunteer-driven organization which compiles detalied data about covid19. 
- [Google social mobility reports](https://www.google.com/covid19/mobility/) - Aggregated cellphone data for countries and regions around the world. We use this data as a measure of social distancing in each country. 

## Running the code and viewing results

You can see the results of our predictions in the [whitepaper's](https://github.com/anag004/covid-model/blob/master/ihme/whitepaper.pdf) results section or in the [Jupyter Notebook](https://github.com/anag004/covid-model/blob/master/SIRD_model.ipynb). The `utils.py` file contains many functions which are used in the notebook. Refer to the docstrings of the functions is `utils.py` for more information on how to use them. To run the code clone the Git repository and run a jupyter notebook server from within the folder. 

```
git clone https://github.com/anag004/covid-model.git
cd covid-model
jupyter notebook
```

If you have jupyter notebook installed your browser window should open. View the `SIRD_model.ipynb` notebook to view the results and play with the code. In case of any issues/suggestions please consider opening an issue or a pull request. 

## Maintainers

- [anag004](github.com/anag004/)
- [tyagiutkarsh](github.com/tyagiutkarsh/)

## License
See the [LICENSE](https://github.com/anag004/covid-model/blob/master/LICENSE.md) file for license rights and limitations (MIT).
