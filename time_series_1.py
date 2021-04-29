import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose

# pd.set_option('display.float_format', lambda x: '%0.2' % x)
warnings.filterwarnings("ignore")

### Dataset ###
"""
Global Land and Ocean-and-Land Temperatures (GlobalTemperatures.csv):
Date: starts in 1750 for average land temperature and 1850 for max and min land temperatures and global ocean and land 
temperatures 
LandAverageTemperature: global average land temperature in celsius LandAverageTemperatureUncertainty: 
the 95% confidence interval around the average LandMaxTemperature: global average maximum land temperature in celsius
 LandMaxTemperatureUncertainty: the 95% confidence interval around the maximum land temperature LandMinTemperature: 
 global average minimum land temperature in celsius LandMinTemperatureUncertainty: the 95% confidence interval around 
 the minimum land temperature LandAndOceanAverageTemperature: global average land and ocean temperature in celsius 
 LandAndOceanAverageTemperatureUncertainty: the 95% confidence interval around the global average land and ocean 
 temperature Other files include:
Global Average Land Temperature by Country (GlobalLandTemperaturesByCountry.csv) Global Average Land Temperature by 
State (GlobalLandTemperaturesByState.csv) Global Land Temperatures By Major City (GlobalLandTemperaturesByMajorCity.csv)
Global Land Temperatures By City (GlobalLandTemperaturesByCity.csv)
The raw data comes from the Berkeley Earth data page
"""
data = pd.read_csv("Medium_article/GlobalLandTemperaturesByState.csv")
data.head()

y = data.copy()
y.dtypes

y.rename(columns={"dt": "Date", "AverageTemperature": "Avg_Temp", "AverageTemperatureUncertainty": "confidence_interval_temp"}, inplace=True)
y.head()

y["Date"] = pd.to_datetime(y["Date"])


y[['Country', "Avg_Temp"]].groupby(["Country"]).mean().sort_values("Avg_Temp")
# it would be more logical to fill in the time series with previous or next values
y = y.fillna(y.bfill())
# Since we have a monthly data, for example, we filled the first day of the 6th month with the first day of the 7th month.
y[['Country', "Avg_Temp"]].groupby(["Country"]).mean().sort_values("Avg_Temp")
#y['Date']
# Creating new date variables
y['Year'] = y['Date'].dt.year
y['Month'] = y['Date'].dt.month
y['Day'] = y['Date'].dt.day

# In the US, I downloaded the data of Florida in order to operate only on Florida's data.
florida = pd.read_csv('Medium_article/florida_file_.csv')
florida.head()
florida.tail(5)
florida["Date"] = pd.to_datetime(florida["Date"])
florida.set_index('Date', inplace = True)

plt.figure(figsize = (6,4))
sns.lineplot(x = 'Year', y = 'Avg_Temp', data = florida)
plt.show()

florida = florida[["Avg_Temp"]]
florida = florida["Avg_Temp"].resample('MS').mean()

train = florida[:"1994-12-01"]
len(train)
test = florida["1995-01-01":]
len(test)
# In this step we are doing a train test split.




def ts_decompose(y, model="additive", stationary=False):
    from statsmodels.tsa.seasonal import seasonal_decompose
    result = seasonal_decompose(y, model=model)
    fig, axes = plt.subplots(4, 1, sharex=True, sharey=False)
    fig.set_figheight(10)
    fig.set_figwidth(15)

    axes[0].set_title("Decomposition for " + model + " model")
    axes[0].plot(y, 'k', label='Original ' + model)
    axes[0].legend(loc='upper left')

    axes[1].plot(result.trend, label='Trend')
    axes[1].legend(loc='upper left')

    axes[2].plot(result.seasonal, 'g', label='Seasonality & Mean: ' + str(round(result.seasonal.mean(), 4)))
    axes[2].legend(loc='upper left')

    axes[3].plot(result.resid, 'r', label='Residuals & Mean: ' + str(round(result.resid.mean(), 4)))
    axes[3].legend(loc='upper left')
    plt.show()

    if stationary:
        print("HO: Series Is Not Stationary.")
        print("H1: Series Is Stationary..")
        p_value = sm.tsa.stattools.adfuller(y)[1]
        if p_value < 0.05:
            print(F"Result: Series Is Stationary. ({p_value}).")
        else:
            print(F"Result: Series Is Not Stationary. ({p_value}).")
for model in ["additive", "multiplicative"]:
    ts_decompose(florida, model, True)


# showing trend and seasonality

### Forecasting with Single Exponential Smoothing ###
# It can be used in stationary series. Not used if there is trend and seasonality

### Stationarity test (Dickey-Fuller Test) ###

def is_stationary(y):
    print('H0: Series Is Not Stationary')
    print("H1: Series Is Stationary")
    p_value = sm.tsa.stattools.adfuller(y)[1]
    if p_value < 0.05:
        print("Result is stationary",p_value)
    else:
        print("Series is not stationary", p_value)
is_stationary(florida)


ses_model = SimpleExpSmoothing(train).fit(smoothing_level = 0.5)
y_pred = ses_model.forecast(225)
train.plot(title="Single Exponential Smoothing")
test.plot()
y_pred.plot()
plt.show()
mean_absolute_error(test, y_pred)


# The green line shown in the graph is the estimate of our model,
# while the orange line is the actual values.
# At this stage, we could not predict trend or seasonality.

# These functions find parameters using the max log likelihood likelihood method.

# Now let's create the grid and make it alphaless, optimize it with the boothforce method.
def optimize_ses(train,alphas, step=225):
    for alpha in alphas:
        ses_model = SimpleExpSmoothing(train).fit(smoothing_level=alpha)
        y_pred = ses_model.forecast(step)
        mae = mean_absolute_error(test, y_pred)
        print("alpha", round(alpha,2), "mae:", round(mae,4))
alphas = np.arange(0.01, 1,0.01)
optimize_ses(train, alphas)

### Final Ses Model ###
ses_model = SimpleExpSmoothing(train).fit(smoothing_level=0.14)
y_pred = ses_model.forecast(225)
train["1985":].plot(title='Single Exponential Smoothing')
test.plot()
y_pred.plot()
plt.show()
mean_absolute_error(test, y_pred)

# We cannot accept this model because we cannot catch the trend and seasonality.


### Forecasting with double exponential smoothing ###

# DES = Level + Trend
des_model = ExponentialSmoothing(train, trend="add").fit(smoothing_level=0.5,
                                                         smoothing_slope=0.5)
y_pred = des_model.forecast(225)
train["1985":].plot(title="Double Exponential Smoothing")
test.plot()
y_pred.plot()
plt.show()
mean_absolute_error(test, y_pred)

# We caught the trend but we could not predict how long the trend will
# continue and the trend has always continued.

###############################################
### Optimizing Double Exponential Smoothing ###
###############################################

def optimize_des(train, alphas, betas, step = 225):
    for alpha in alphas:
        for beta in betas:
            des_model = ExponentialSmoothing(train, trend = 'add').fit(smoothing_level=alpha,
                                                                       smoothing_slope= beta)
            y_pred = des_model.forecast(step)
            mae = mean_absolute_error(test, y_pred)
            print("alpha:", round(alpha, 2), "beta:", round(beta, 2), "mae:", round(mae, 4))


alphas = np.arange(0.01, 1, 0.05)
betas = np.arange(0.01, 1, 0.05)

optimize_des(train, alphas, betas)

def optimize_des(train, alphas, betas, step=225):
    print("Optimizing parameters...")
    results = []
    for alpha in alphas:
        for beta in betas:
            des_model = ExponentialSmoothing(train, trend="add").fit(smoothing_level=alpha,
                                                                     smoothing_slope=beta)
            y_pred = des_model.forecast(step)
            mae = mean_absolute_error(test, y_pred)
            results.append([round(alpha, 2), round(beta, 2), round(mae, 6)])
    results = pd.DataFrame(results, columns=["alpha", "beta", "mae"]).sort_values("mae")
    print(results)

optimize_des(train, alphas, betas)
# 5    0.01  0.01   4.32

### Final DES Model ###

final_des_model = ExponentialSmoothing(train, trend="add").fit(smoothing_level=0.06,
                                                               smoothing_slope=0.01)
y_pred = final_des_model.forecast(225)
train["1985":].plot(title="Double Exponential Smoothing")
test.plot()
y_pred.plot()
plt.show()
mean_absolute_error(test, y_pred)



## Forecasting with Triple Exponential Smoothing (HOLT WINTERS) ###
tes_model = ExponentialSmoothing(train,
                                 trend='add',
                                 seasonal="add",
                                 seasonal_periods=12).fit(smoothing_level=0.5,
                                                          smoothing_slope=0.5,
                                                          smoothing_seasonal=0.5)
y_pred = tes_model.forecast(225)
train["1985":].plot(title = "Triple Exponential Smoothing")
test.plot()
y_pred.plot()
plt.show()
mean_absolute_error(test, y_pred)

### Optimizing Triple Exponential Smoothing ###
import itertools
alphas = betas = gammas = np.arange(0.01, 1, 0.05)
abg = list(itertools.product(alphas, betas, gammas))
abg[0][2]

def optimize_tes(train, abg, step=225):
    print("Optimizing parameters...")
    results = []
    for comb in abg:
        tes_model = ExponentialSmoothing(train, trend="add",
                                         seasonal="add",
                                         seasonal_periods=12).\
            fit(smoothing_level=comb[0],
                smoothing_slope=comb[1],
                smoothing_seasonal=comb[2])

        y_pred = tes_model.forecast(step)
        mae = mean_absolute_error(test, y_pred)

        print([round(comb[0], 2), round(comb[1], 2), round(comb[2], 2), round(mae, 2)])

        results.append([round(comb[0], 2), round(comb[1], 2), round(comb[2], 2), round(mae, 2)])
    results = pd.DataFrame(results, columns=["alpha", "beta", "gamma", "mae"]).sort_values("mae")
    print(results)

alphas = betas = gammas = np.arange(0.01, 1, 0.05)
abg = list(itertools.product(alphas, betas, gammas))

optimize_tes(train, abg)
#################################
# Final TES Model
#################################
final_tes_model = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=12).\
            fit(smoothing_level=0.06, smoothing_slope=0.06, smoothing_seasonal=0.06)
y_pred = final_tes_model.forecast(225)
train["1985":].plot(title="Triple Exponential Smoothing")
test.plot()
y_pred.plot()
plt.show()
mean_absolute_error(test, y_pred)

