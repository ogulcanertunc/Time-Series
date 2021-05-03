import itertools
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.tsa.api as smt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
pd.set_option("max_columns",None)
warnings.filterwarnings("ignore")




florida = pd.read_csv('florida_file_.csv')
florida.head()
florida.tail(5)
florida["Date"] = pd.to_datetime(florida["Date"])
florida.set_index('Date', inplace = True)

florida = florida[["Avg_Temp"]]
florida = florida["Avg_Temp"].resample('MS').mean()
florida = florida.fillna(florida.bfill())


train = florida[:"1994-12-01"]
len(train)
test = florida["1995-01-01":]
len(test)

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
        print("HO: Series is not Stationary.")
        print("H1: Series is Stationary.")
        p_value = sm.tsa.stattools.adfuller(y)[1]
        if p_value < 0.05:
            print(F"Result: Series is Stationary ({p_value}).")
        else:
            print(F"Result: Series is not Stationary ({p_value}).")

for model in ["additive","multiplicative"]:
    ts_decompose(florida,model, True)

####################
### ARIMA Model ####
####################

arima_model = ARIMA(train, order = (1,0,1)).fit(disp=0)
arima_model.summary()

y_pred= arima_model.forecast(225)[0]
mean_absolute_error(test, y_pred)
# 4.39981

arima_model.plot_predict(dynamic=False)
plt.show()

train["1985":].plot(legend=True, label = 'TRAIN')
test.plot(legend=True, label = 'TEST', figsize = (6,4))
pd.Series(y_pred, index=test.index).plot(legend=True, label = 'Prediction')
plt.title("Train, Test and Predicted Test")
plt.show()
# Our predictions don't look very well

##########################
### ARIMA Model Tuning ###
##########################

# Statistical Consideration of Model Grade Selection #
######################################################

# 1. Determining the model grade according to ACF & PACF Graphs
# 2. Determining the Model Rank According to AIC Statistics

##################################################################
### Determining the model grade according to ACF & PACF Graphs ###
##################################################################

def acf_pacf(y, lags=30):
    plt.figure(figsize=(12, 7))
    layout = (2, 2)
    ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
    acf_ax = plt.subplot2grid(layout, (1, 0))
    pacf_ax = plt.subplot2grid(layout, (1, 1))
    y.plot(ax=ts_ax)

    # Durağanlık testi (HO: Seri Durağan değildir. H1: Seri Durağandır.)
    p_value = sm.tsa.stattools.adfuller(y)[1]
    ts_ax.set_title('Time Series Analysis Plots\n Dickey-Fuller: p={0:.5f}'.format(p_value))
    smt.graphics.plot_acf(y, lags=lags, ax=acf_ax)
    smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax)
    plt.tight_layout()
    plt.show()

acf_pacf(florida)

# y, yt-1, yt-2
# In the output, we see that autoregression makes sense because they are out of the blue area.


# the width of the acf graph decreases with the lags, and at the same time the pacf graph is also after a certain delay
# If it cuts sharply, it means it's ARMA model

df_diff = florida.diff()
df_diff.dropna(inplace=True)

acf_pacf(df_diff)

### Determining Model Rank According to AIC & BIC Statistics ###
################################################################
# Generation of combinations of p and q

p = d = q = range(0, 4)
pdq = list(itertools.product(p, d, q))

def arima_optimizer_aic(train, orders):
    best_aic, best_params = float("inf"), None

    for order in orders:
        try:
            arma_model_result = ARIMA(train, order).fit(disp=0)
            aic = arma_model_result.aic
            if aic < best_aic:
                best_aic, best_params = aic, order
            print('ARIMA%s AIC=%.2f' % (order, aic))
        except:
            continue
    print('Best ARIMA%s AIC=%.2f' % (best_params, best_aic))
    return best_params

best_params_aic = arima_optimizer_aic(train, pdq)

### Tuned Model ###
###################
arima_model = ARIMA(train, best_params_aic).fit(disp=0)
y_pred = arima_model.forecast(225)[0]
mean_absolute_error(test, y_pred)
# 1.4470

train["1985":].plot(legend=True, label="TRAIN")
test.plot(legend=True, label="TEST", figsize=(6, 4))
pd.Series(y_pred, index=test.index).plot(legend=True, label="PREDICTION")
plt.title("Train, Test and Predicted Test")
plt.show()


############################################################################
### SARIMA(p, d, q): (Seasonal Autoregressive Integrated Moving-Average) ###
############################################################################
from statsmodels.tsa.statespace.sarimax import SARIMAX

train = florida[:"1994-12-01"]
len(train)
test = florida["1995-01-01":]
len(test)
val = train["1991-01-01":]
len(val)

##########################################
### Structural Analysis of Time Series ###
##########################################

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
        print("HO: Series is not Stationary.")
        print("H1: Series is Stationary.")
        p_value = sm.tsa.stattools.adfuller(y)[1]
        if p_value < 0.05:
            print(F"Result: Series is stationary ({p_value}).")
        else:
            print(F"Result: Series is not stationary ({p_value}).")
for model in ["additive", "multiplicative"]:
    ts_decompose(florida, model, True)

### Model ###

model = SARIMAX(train, order=(1,0,1), seasonal_order=(0,0,0,12))
sarima_model = model.fit(disp=0)

### Validation Error ###
########################
len(val)
# 48
len(train)
# 3014
len(test)
# 225
pred = sarima_model.get_prediction(start = pd.to_datetime('1991-01-01'),dynamic=False)
pred_ci = pred.conf_int()

y_pred = pred.predicted_mean
mean_absolute_error(val, y_pred)
# 1.9192

### Validation Predictions Visualization ###
############################################

ax = train["1985":].plot(label='TRAIN')
pred.predicted_mean.plot(ax=ax, label='VALIDATION FORECASTS', alpha=.7)
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)

ax.set_xlabel('Date')
ax.set_ylabel('Temperature')
plt.legend()
plt.show()

### Test Error ###
##################
y_pred_test = sarima_model.get_forecast(steps=225)
pred_ci = y_pred_test.conf_int()
y_pred = y_pred_test.predicted_mean
mean_absolute_error(test, y_pred)
# 17.0051

### Test Predictions Visualization ###
############################################
ax = train["1985":].plot(label='TRAIN')
test.plot(legend=True, label="TEST")
y_pred_test.predicted_mean.plot(ax=ax, label='TEST FORECASTS', alpha=.7)
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)

ax.set_xlabel('Date')
ax.set_ylabel('Temperature')
plt.legend()
plt.title("Forecast vs Real for Test")
plt.show()

### MODEL Tuning ###
####################

p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

def sarima_optimizer_aic(train, pdq, seasonal_pdq):
    best_aic, best_order, best_seasonal_order = float("inf"), float("inf"), None
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                sarimax_model = SARIMAX(train, order=param, seasonal_order=param_seasonal)
                results = sarimax_model.fit(disp=0)
                aic = results.aic
                if aic < best_aic:
                    best_aic, best_order, best_seasonal_order = aic, param, param_seasonal
                print('SARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, aic))
            except:
                continue
    print('SARIMA{}x{}12 - AIC:{}'.format(best_order, best_seasonal_order, best_aic))
    return best_order, best_seasonal_order

best_order, best_seasonal_order = sarima_optimizer_aic(train, pdq, seasonal_pdq)

###################
### Final Model ###
###################

model = SARIMAX(train, order=best_order, seasonal_order=best_seasonal_order)
sarima_final_model = model.fit(disp=0)


###############################
### Final Model Test Error ###
###############################

y_pred_test = sarima_final_model.get_forecast(steps=225)
pred_ci = y_pred_test.conf_int()
y_pred = y_pred_test.predicted_mean
mean_absolute_error(test, y_pred)
# 1.013578872841977

### Visualization of Final Model ###
####################################

ax = florida["1985":].plot(label='TRAIN')
test.plot(legend=True, label="TEST")
y_pred_test.predicted_mean.plot(ax=ax, label='TEST FORECASTS', alpha=.7)
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)

ax.set_xlabel('Date')
ax.set_ylabel('Temperature')
plt.legend()
plt.show()

###############################################
#### Analyzing Statistical outputs of Model ###
###############################################

sarima_final_model.plot_diagnostics(figsize=(15, 12))
plt.show()

############################################
### BONUS: SARIMA OPTIMIZER based on MAE ###
############################################
def fit_model_sarima(train, val, pdq, seasonal_pdq):
    sarima_model = SARIMAX(train, order=pdq, seasonal_order=seasonal_pdq).fit(disp=0)
    y_pred_val = sarima_model.get_forecast(steps=48)
    y_pred = y_pred_val.predicted_mean
    return mean_absolute_error(val, y_pred)

fit_model_sarima(train, val, (0, 1, 0), (0, 0, 0, 12))

p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]


def sarima_optimizer_mae(train, val, pdq, seasonal_pdq):
    best_mae, best_order, best_seasonal_order = float("inf"), float("inf"), None
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mae = fit_model_sarima(train, val, param, param_seasonal)
                if mae < best_mae:
                    best_mae, best_order, best_seasonal_order = mae, param, param_seasonal
                print('SARIMA{}x{}12 - MAE:{}'.format(param, param_seasonal, mae))
            except:
                continue
    print('SARIMA{}x{}12 - MAE:{}'.format(best_order, best_seasonal_order, best_mae))
    return best_order, best_seasonal_order


best_order, best_seasonal_order = sarima_optimizer_mae(train, val, pdq, seasonal_pdq)

model = SARIMAX(train, order=best_order, seasonal_order=best_seasonal_order)
sarima_final_model = model.fit(disp=0)

y_pred_test = sarima_final_model.get_forecast(steps=225)
pred_ci = y_pred_test.conf_int()
y_pred = y_pred_test.predicted_mean
mean_absolute_error(test, y_pred)
# 0.92


ax = florida["1985":].plot(label='TRAIN')
test.plot(legend=True, label="TEST")
y_pred_test.predicted_mean.plot(ax=ax, label='TEST FORECASTS', alpha=.7)
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)

ax.set_xlabel('Date')
ax.set_ylabel('Temperature')
plt.legend()
plt.show()
