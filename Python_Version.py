import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import datetime as dt
import sklearn.metrics as metrics
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from statistics import stdev
from statistics import mean
from math import sqrt
from scipy import stats
from tbats import BATS, TBATS

#create method to remove duplicates from lists (may be useful later when removing outlier indexes)
def remove_duplicates(original_list, second_list):
    final_indeces = original_list
    for i in second_list:
        if i not in final_indeces:
            final_indeces.append(i)
    return final_indeces

#method created to get standard summary statistics for models
def regression_results(x_predictors, y_true, y_pred):

    # Regression metrics
    explained_variance=metrics.explained_variance_score(y_true, y_pred)
    mean_absolute_error=metrics.mean_absolute_error(y_true, y_pred)
    mse=metrics.mean_squared_error(y_true, y_pred)
    median_absolute_error=metrics.median_absolute_error(y_true, y_pred)
    r2=metrics.r2_score(y_true, y_pred)

    #p-value
    x2 = sm.add_constant(x_predictors)
    est = sm.OLS(y_true, x2)
    est_final = est.fit()

    print('explained_variance: ', round(explained_variance,4))
    print('r2: ', round(r2,4))
    print('MAE: ', round(mean_absolute_error,4))
    print('MSE: ', round(mse,4))
    print('RMSE: ', round(np.sqrt(mse),4))

    print(est_final.summary())

#import all relevant datasets
#Note, we initially want to get the date values in the train set to be set to the datetime data type
sales_train = pd.read_csv('D:\Projects\Kaggle_Projects\Future_Sales_Prediction\sales_train_set.csv',',', parse_dates=["date"])
sales_test = pd.read_csv('D:\Projects\Kaggle_Projects\Future_Sales_Prediction\sales_test_set.csv',',')
items = pd.read_csv('D:\Projects\Kaggle_Projects\Future_Sales_Prediction\items.csv',',')
shops = pd.read_csv('D:\Projects\Kaggle_Projects\Future_Sales_Prediction\shops.csv',',')
item_categories = pd.read_csv('D:\Projects\Kaggle_Projects\Future_Sales_Prediction\item_categories.csv',',')

df_train = pd.merge(sales_train, items[['item_category_id', 'item_id']], on = 'item_id')
df_test = pd.merge(sales_test, items[['item_category_id', 'item_id']], on = 'item_id')

#manipulate datetime data as it causes plotting issues
df_train["date"] = pd.to_datetime(df_train["date"], unit = 'D')


#basic initial inspection to get a feel for the data
print(df_train.head())
print('')
print(pd.DataFrame.describe(df_train))

#after initial inspection of the data there are some clear outlier datapoints
#first we will try removing them to see how our predictions fare
#I arbitrarily chose 10 standard deviations from the mean
outlier_indexes = list(df_train[df_train['item_price'].gt(mean(df_train['item_price'])+(10*stdev(df_train['item_price'],mean(df_train['item_price']))))].index)
outlier_indexes2_electric_boogaloo = list(df_train[df_train['item_cnt_day'].gt(mean(df_train['item_cnt_day'])+(10*stdev(df_train['item_cnt_day'],mean(df_train['item_cnt_day']))))].index)
full_list = remove_duplicates(outlier_indexes, outlier_indexes2_electric_boogaloo)
print('')
print(len(full_list))
df_train = df_train.drop(full_list)

#split to train and test data for training
x_train, x_test, y_train, y_test = train_test_split(df_train.drop('item_cnt_day', axis=1), df_train.item_cnt_day,test_size=.25)

print('')
print(x_train.head())
print('')
print(x_train.shape)


#basic line scatter plot with simple regression
#sns.lmplot(y_train,x_train['item_price'], fit_reg=False)
#sns.lmplot(x='item_cnt_day',y='item_price', data = df_train)
#plt.show()

#Seaborn requires one line for data and one line for indexing so quick coersion for the data here so we can plot
coerced_df = pd.DataFrame({
    'date': df_train["date"],
    'item_price' : df_train['item_price'],
    'item_cnt_day' : df_train['item_cnt_day']
})

#coerced data plot was useless because I did not think about how scale of data was different
#sns.set_style("darkgrid")
#sns.lineplot(x="date",y="value", hue = 'variable', data=pd.melt(coerced_df, ['date']))

#creating two linegraphs on top of each other with date as x axis and separate y limits based on their own data
#g = sns.PairGrid(df_train, y_vars = ['item_price','item_cnt_day'], x_vars = ['date'])
#g.map(sns.lineplot)
#plt.show()


#x_scaled = preprocessing.scale(x_train.drop(columns = 'date'))
#print(x_scaled.head())
#sns.distplot(x_scaled["item_price"])
#plt.show()

#for regression reasons, we are converting dates to ordinal integers
x_train['date'] = pd.to_datetime(x_train['date'])
x_train['date'] = x_train['date'].map(dt.datetime.toordinal)
x_test['date'] = pd.to_datetime(x_test['date'])
x_test['date'] = x_test['date'].map(dt.datetime.toordinal)

#Scale the data and set up scaled model for easier transformation
#scaler = preprocessing.StandardScaler().fit(x_train)
#x_train_scaled = scaler.transform(x_train)
#x_test_scaled = scaler.transform(x_test)

lin_model = LinearRegression()
lin_model_fit = lin_model.fit(x_train,y_train)
lin_model_pred = lin_model_fit.predict(x_test)

regression_results(x_test, y_test, lin_model_pred)


#lin_model_scaled = LinearRegression()
#lin_model_fit_scaled = lin_model_scaled.fit(x_train_scaled,y_train)
#lin_model_pred_scaled = lin_model_fit_scaled.predict(x_test_scaled)

#regression_results(x_test_scaled, y_test, lin_model_pred_scaled)

##Ultimately proved not very helpful to scale data

estimator = TBATS(seasonal_periods=[14,30.5])
tbats_model = estimator.fit(y_train)
tbats_pred = tbats_model.forecast(steps=1)