import os
import pandas as pd     
import pandas_profiling
import numpy as np     
from matplotlib import pyplot
import matplotlib.pyplot as plt  
import datetime  
from pandas import DataFrame
from pandas import TimeGrouper  
from pandas import Series        
%matplotlib inline
import time
from datetime import timedelta
import warnings                   
warnings.filterwarnings("ignore")
from pandas import Series
from sklearn.metrics import mean_squared_error
from math import sqrt 
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.api import SimpleExpSmoothing
from statsmodels.tsa.api import ExponentialSmoothing
from statsmodels.tsa.api import Holt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.arima_model import ARIMA
from fbprophet import Prophet
from statsmodels.tools.sm_exceptions import ValueWarning
import pandas as pd
import pymc3 as pm
from pmprophet.model import PMProphet
from statsmodels.tsa.arima_model import ARMA

import warnings
warnings.filterwarnings('ignore')


'''
Splitting the data intro train and test to validate our model. Default split of 90:10 is set for initial analysis. 
A ranger fucntion is defined to enumerate uniform step values between the provided range
Seasonality decomposition graph to understand the components of time series

Functions to convert date time objects and group by year, month 
Function to split datetime object into corresponding year, month and day
Aggregate our taget variable based on year, month to understand the pattern of time series
          
The difference of a time series is the series of changes from one period to the next. Differecing is done 
'''

#Outlier detection and imputing null values
def outlier_detection(data, column, types):
    data_copy = data.copy()
    quartile_1, quartile_3 = np.percentile(data_copy[column], [25, 75])
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    data_copy.loc[data_copy[column] > upper_bound , column] = np.nan
    data_copy.loc[data_copy[column] < lower_bound , column] = np.nan
    data_copy[column].fillna(method = types, inplace = True)
    return data_copy

# define a dataset with a linear trend
def plot_difference(data, column, interval):
    plt.figure(figsize= (20,10))
    plt.title('Transformed distribution')
    temp = data.copy()
    y = temp[column]
    y_shift = temp[column].shift(interval)
    temp['date'] = pd.to_datetime(temp['date'])
    start = temp['date'][0]
    dates = [start + datetime.timedelta(n) for n in range(len(y))]
    plt.plot(dates,y, label = 'Actual data')
    plt.plot(dates, y_shift, label = 'Shifted data')
    plt.xlabel("Years", fontsize = 15)
    plt.ylabel(column, fontsize = 15)
    plt.legend(loc='upper left', prop={'size': 20} )
    plt.tick_params(axis='both', which='major', labelsize=15)
    plt.show()

#Stationarity test using DF test
def test_stationarity(data, column):
    print ('--------------------------------Results of Dickey-Fuller Test---------------------------------------')
    stationary = data[column].values
    result = adfuller(stationary)
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))
    for key,value in result[4].items():
            if result[0]>value:
                print("\n The graph is non stationary since ADF test statistic (%.3f) is not less than 1 percent critical values (%.3f)" %( result[0],value))
                break;
            else: 
                print("\n The graph is stationary since ADF test statistic (%.3f) is less than critical values (%.3f)" %(result[0],value))
                break;  
    seasonality_decomp(data, column) 

def create_df():
#create result data frames to save results    
    col_names =  ['Column_name','Method', 'RMSE', 'Time']
    result = pd.DataFrame(columns = col_names)
    res_plot = pd.DataFrame()
    return result,res_plot

def ranger(x,y,j):
#customized range function with customizable step value    
    while x<y:
        yield x
        x += j    

def seasonality_decomp(data, column):
#Seasonality decomposition technique to determine trend, seasonality and residual in the data    
    data = difference(data, column, 1)
    decomposition = seasonal_decompose(data[column].values, freq = 24)
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid
    print('Time series into respective components - Trend, Seasonality and Residuals')
    plt.figure(figsize=(20,10))
    plt.subplot(411)
    plt.title('Decomposition plot of time series', fontsize = 25)
    plt.plot(data[column], label='Original', color  = 'r')
    plt.legend(loc='upper left', prop={'size': 15})
    plt.subplot(412)
    plt.plot(trend, label='Trend', color  = 'magenta')
    plt.legend(loc='upper left', prop={'size': 15})
    plt.subplot(413)
    plt.plot(seasonal,label='Seasonality', color = 'green')
    plt.legend(loc='upper left', prop={'size': 15})
    plt.subplot(414)
    plt.plot(residual, label='Residual')
    plt.legend(loc='upper left', prop={'size': 15})
    plt.xlabel('Date', fontsize=20,labelpad = 20)
    plt.show()         

def difference(data, column, interval):
#differencing function to shift data with custom interval param    
    df_copy = data.copy()
    df_copy[column] = df_copy[column] - df_copy[column].shift(interval)
    df_copy[column] = df_copy[column].fillna(method = 'bfill')
    return df_copy   

def inverse_difference(last_ob, value):
#inverse the shifted data    
    return value + last_ob

def log_transform(data,column):
#log transformation of the data    
    df_copy = data.copy()
    df_copy[column] = np.log(df_copy[column] + 1)
    return df_copy

def inverse_transform(data,column):
#inverse log transform of the data    
    df_copy = data.copy()
    df_copy[column] = np.expm1(df_copy[column])
    return df_copy[column]

def split_data(data, column):
#split the data into train and test dataframes  
    df_copy = data.copy()
    df_copy = df_copy.set_index('date') 
    num =(int(df_copy.shape[0]) - 7)/int(df_copy.shape[0])
    size = int(df_copy.shape[0] * num)
    train, test = df_copy[0:size], df_copy[size:]
    train = pd.DataFrame(train, columns= [column])
    test = pd.DataFrame(test, columns= [column])
    #print('Size of training data :', train.shape)
    #print('Size of test data :', test.shape)
    return train,test
        
def acf_pacf(data, column):
#Plot autocorreclation and partial autocorrelation to determine p,d,q values for arima
    series = data[column].values
    plt.figure(figsize=(20,12))
    plt.subplot(411)
    plot_acf(series, lags= 20, ax=plt.gca(), use_vlines= True)
    plt.xlabel('Lags', fontsize =12)
    plt.xticks(np.arange(0, 21))
    
    plt.axhline(y=0,linestyle='--',color='gray')
    plt.axhline(y=-1.96/np.sqrt(len(series)),linestyle='--',color='gray')
    plt.axhline(y=1.96/np.sqrt(len(series)),linestyle='--',color='gray')
    
    plt.subplot(412)
    plot_pacf(series, lags=20, ax=plt.gca())
    plt.tight_layout()
    plt.xlabel('Lags', fontsize =12)
    plt.xticks(np.arange(0, 21))
    
    plt.axhline(y=0,linestyle='--',color='gray')
    plt.axhline(y=-1.96/np.sqrt(len(series)),linestyle='--',color='gray')
    plt.axhline(y=1.96/np.sqrt(len(series)),linestyle='--',color='gray')    
    plt.show()
    
def test_plt(img_size, test, pred, column, title):
#a custom plot function to be used in all the future methods   
    #import pdb
    #pdb.set_trace()
    plt.figure(figsize= img_size)
    plt.title(title, fontsize = 25)
    plt.xlabel('Date', fontsize = 18)
    plt.ylabel(column, fontsize = 18)
    plt.plot(test[column], label='Test_data')
    plt.plot(pred, label= 'Predictions')
    plt.xticks(rotation=90)
    plt.legend(loc='upper left',prop={'size': 15})
    plt.show()
   # else:
   #     plt.figure(figsize= img_size)
   #     plt.title(title, fontsize = 25)
   #     plt.xlabel('Date', fontsize = 18)
   #     plt.ylabel(column, fontsize = 18)
   #     plt.plot(test[column], label='Test_data')
   #     plt.plot(pred, label= 'Predictions')
   #     plt.xticks(rotation=90)
   #     plt.legend(loc='upper left',prop={'size': 15})
   #     plt.savefig(folder_name + "_" + title + "_" + column)
   #     plt.close()
   #     
#--------------------------------------------------------------------------------------------------------------------------        


def build_model(data, column, method, interval = 1, ma_limit = 30 , se = 3.0, hw_period =7, smoothing =0.01, slope =0.1, p_values = range(0,4), d_values = range(0,2), q_values = range(0,2)):
# Build model function which takes in default parameters for each method, user can change each parameter values as per business needs

    train,test = split_data(data, column)
    train[column] = log_transform(train, column)
    if method == 'moving_average':
        method_wrapper(moving_average,train,test,column,ma_limit)
    elif method == 'simple_exponential_smoothing':
        method_wrapper(simple_exponential_smoothing,train,test,column,se)
    elif method == 'holt_winters':
        method_wrapper(holt_winters,train,test,column,hw_period)
    elif method == 'holts_linear':
        method_wrapper(holts_linear,train,test,column, smoothing, slope)
    elif method == 'arima':
        method_wrapper(arima,train,test,column,p_values, d_values, q_values)
    elif method == 'sarimax':
        method_wrapper(sarimax,train,test,column, p_values, d_values, q_values)
    elif method == 'fbprophet':
        method_wrapper(fb_prophet, train, test, column)
    elif method == 'pmprophet':
        method_wrapper(pm_prophet, train, test, column)      
        
def method_wrapper(method, *vargs):
#a wrapper to call a function and pass respective parameters    
    method(*vargs)

def moving_average(train, test, column, ma_limit):
#moving average where ma_limit is the period value
    y_hat_avg = test.copy()
    rmse_ma = float("inf")                                   
    start = time.time()                     
    y_hat_avg['moving_avg_forecast'] = train[column].rolling(ma_limit).mean().iloc[-1]
    y_hat_avg['transformed_values'] = inverse_transform(y_hat_avg,'moving_avg_forecast')
    rmse_ma = sqrt(mean_squared_error(y_hat_avg[column], y_hat_avg['transformed_values']))
    end = time.time()       
    time_ma = end - start    
    print('\n Total RMSE of the Naive approach - moving average model : %.3f' % rmse_ma)
    result.loc[len(result)] = [column, 'Naive forecast - Moving average', rmse_ma, time_ma] 
    #test_plt((20,10), test, predictions, column, 'Moving averge forecast')
    result_plots['moving_average'] = y_hat_avg['transformed_values']

def simple_exponential_smoothing(train, test, column, se):
#simple exponential smoothing technique where se is the smoothing level    
    y_hat_avg = test.copy()
    rmse_se = float("inf")
    start = time.time()
    for p in ranger(0,se,0.01):
        model_ses = SimpleExpSmoothing(np.asarray(train[column])).fit(smoothing_level=0.1,optimized=False)
        y_hat_avg['simple_exponential_smoothing']= model_ses.forecast(len(test))
        y_hat_avg['transformed_values'] = inverse_transform(y_hat_avg,'simple_exponential_smoothing')
        rms = sqrt(mean_squared_error(y_hat_avg[column], y_hat_avg['transformed_values']))
        if rms < rmse_se:
            rmse_se = rms
    end = time.time()            
    time_se = end - start
    print('\n Total RMSE of the simple exponential smoothing model : %.3f ' % rmse_se) 
    result.loc[len(result)] = [column, 'Simple exponential smooting', rmse_se, time_se] 
    #test_plt((20,10), test, predictions, column, 'Simple exponential smoothing')
    result_plots['simple_exponential_smoothing'] = y_hat_avg['transformed_values']  

def holt_winters(train, test, column, hw_period):
#holt_winters technique where hw_period is the seasonal periods
    y_hat_avg = test.copy()
    start = time.time()
    model_hw = ExponentialSmoothing(np.asarray(train[column]), seasonal_periods = hw_period, trend='add', seasonal='add', damped = True).fit()
    y_hat_avg['holt_winters']= model_hw.forecast(len(test))
    y_hat_avg['transformed_values'] = inverse_transform(y_hat_avg,'holt_winters')
    rmse_hw = sqrt(mean_squared_error(y_hat_avg[column], y_hat_avg['transformed_values']))
    end = time.time()            
    time_hw = end - start
    print('\n Total RMSE of the holt winters model : %.3f ' % rmse_hw) 
    result.loc[len(result)] = [column, 'Holt winters', rmse_hw, time_hw] 
    #test_plt((20,10), test, predictions, column, 'Simple exponential smoothing')
    result_plots['holt_winters'] = y_hat_avg['transformed_values']

def holts_linear(train, test, column, smoothing, slope):
#holt linear technique with smoothing and slope as parameters
    y_hat_avg = test.copy()
    start = time.time()
    model_hl = Holt(np.asarray(train[column])).fit(smoothing_level = smoothing, smoothing_slope = slope)
    y_hat_avg['holt_linear']= model_hl.forecast(len(test))
    y_hat_avg['transformed_values'] = inverse_transform(y_hat_avg,'holt_linear')
    rmse_hl = sqrt(mean_squared_error(y_hat_avg[column], y_hat_avg['transformed_values']))
    end = time.time()            
    time_hl = end - start
    print('\n Total RMSE of the holt linear model : %.3f ' % rmse_hl)  
    result.loc[len(result)] = [column, 'Holts Linear', rmse_hl, time_hl] 
    #test_plt((20,10), test_df, predictions, column, 'Holts Linear')
    result_plots['Holts_linear'] = y_hat_avg['transformed_values']
    
def arima(train, test, column, p_values, d_values, q_values):
#evaluate combinations of p, d and q values for an ARIMA model    
    y_hat_avg = test.copy()
    rmse_arima = float("inf")
    start = time.time()
    for p in p_values:
        for d in d_values:
            for q in q_values:
                arima_order = (p,d,q)
                model_arima = ARIMA(train[column], order = arima_order).fit(disp=0)
                y_hat_avg['arima'] = model_arima.forecast(len(test))[0]
                y_hat_avg['transformed_values'] = inverse_transform(y_hat_avg,'arima')
                rms = sqrt(mean_squared_error(y_hat_avg[column], y_hat_avg['transformed_values']))
                if rms < rmse_arima:
                    rmse_arima = rms
    end = time.time()            
    time_arima = end - start
    print('\n Total RMSE of the ARIMA model : %.3f ' % rmse_arima)  
    result.loc[len(result)] = [column, 'arima', rmse_arima, time_arima] 
    #test_plt((20,10), test_df, predictions, column, 'arima')
    result_plots['arima'] = y_hat_avg['transformed_values']

def sarimax(train, test, column, p_values, d_values, q_values):
#evaluate combinations of p, d and q values for an SARIMAX model      
    y_hat_avg = test.copy()
    rmse_sarimax = float("inf")
    start = time.time()
    for p in p_values:
        for d in d_values:
            for q in q_values:
                sarimax_order = (p,d,q)
                model_sarimax = sm.tsa.statespace.SARIMAX(train[column], order = sarimax_order).fit(disp=0)
                y_hat_avg['sarimax'] = list(model_sarimax.forecast(len(test)))
                y_hat_avg['transformed_values'] = inverse_transform(y_hat_avg,'sarimax')
                rms = sqrt(mean_squared_error(y_hat_avg[column], y_hat_avg['transformed_values']))
                if rms < rmse_sarimax:
                    rmse_sarimax = rms
    end = time.time()            
    time_sarimax = end - start
    print('\n Total RMSE of the SARIMAX model : %.3f ' % rmse_sarimax)  
    result.loc[len(result)] = [column, 'sarimax', rmse_sarimax, time_sarimax] 
    #test_plt((20,10), test_df, predictions, column, 'sarimax')
    result_plots['sarimax'] = y_hat_avg['transformed_values']    

def fb_prophet(train, test, column):
#fbprophet method to solve the time series    
    #data.loc[(data['date'] > '2017-03-01') & (data['date'] < '2017-05-01'), 'credit_cards_added'] = None
    y_hat_avg = test.reset_index().copy()
    train = train.reset_index()
    train['date'] = pd.to_datetime(train['date'])
    train.columns = ['ds', 'y']
    start = time.time()
    m = Prophet(weekly_seasonality=False, interval_width = 0.95 , yearly_seasonality=True, daily_seasonality= True, monthly_seasonality = True )
    #m.add_seasonality(name='weekly', period= 7, fourier_order= 5)
    m.fit(train)
    future = m.make_future_dataframe(periods=7)
    forecast = m.predict(future)
    y_hat_avg['fb_prophet'] = forecast.yhat
    y_hat_avg['transformed_values'] = inverse_transform(y_hat_avg,'fb_prophet')
    rmse_fb = sqrt(mean_squared_error(y_hat_avg[column], y_hat_avg['transformed_values']))
    end = time.time()
    time_fb = end - start
    print('\n The total RMSE using FB prophet : %.3f' % rmse_fb)
    result.loc[len(result)] = [column, 'Fbprophet', rmse_fb, time_fb] 
    #test_plt((20,10), test_df, predictions, column, 'fbprophet')
    result_plots['fb_prophet'] = y_hat_avg['transformed_values']    
    result_plots['test'] = y_hat_avg[column]
    result_plots['date'] = y_hat_avg['date']


def pm_prophet(train,test, column):
    # Fit both growth and intercept
    train = train.reset_index()
    train['date'] = pd.to_datetime(train['date'])
    train.columns = ['ds', 'y']
    start = time.time()
    m = PMProphet(train, growth=True, intercept=True, n_change_points=2, name='model')
    # Add monthly seasonality (order: 3)
    m.add_seasonality(seasonality=30, order=3)
    # Add weekly seasonality (order: 3)
    m.add_seasonality(seasonality=7, order=3)
    # Fit the model (using NUTS, 1e+4 samples and no MAP init)
    m.fit(
        draws=10**4,
        method='NUTS',
        map_initialization=False,
    )
    ddf = m.predict(7, alpha=0.4, include_history=True, plot=True)
    m.plot_components(
        intercept=False,
    )
    pred = inverse_transform(ddf[-7:], 'y_hat')
    rmse_pm = sqrt(mean_squared_error(test, pred))
    end = time.time()
    time_fb = end - start
    print('\n The total RMSE using FB prophet : %.3f' % rmse_pm)
    result.loc[len(result)] = [column, 'pm_prophet', rmse_pm, time_pm] 
    #test_plt((20,10), test_df, predictions, column, 'pmprophet')
    result_plots['pm_prophet'] = pred 
    result.loc[len(result)] = [' ', ' ', ' ', ' ']
 