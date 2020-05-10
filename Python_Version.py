import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from statistics import stdev
from statistics import mean



df_train = pd.read_csv('D:\Projects\Kaggle_Projects\Predict_Future_Sales\sales_train_set.csv',',')
df_test = pd.read_csv('D:\Projects\Kaggle_Projects\Predict_Future_Sales\sales_test_set.csv',',')
items = pd.read_csv('D:\Projects\Kaggle_Projects\Predict_Future_Sales\items.csv',',')
shops = pd.read_csv('D:\Projects\Kaggle_Projects\Predict_Future_Sales\shops.csv',',')

print(df_train.head())
print('')
print(pd.DataFrame.describe(df_train))

#outlier_cutoff =
outlier_indexes = df_train[df_train['item_price'].gt(mean(df_train['item_price'])+(10*stdev(df_train['item_price'],mean(df_train['item_price']))))].index
print(outlier_indexes)
#df_train = df_train.drop([outlier_indexes])

x_train, x_test, y_train, y_test = train_test_split(df_train.drop('item_cnt_day', axis=1), df_train.item_cnt_day,test_size=.25)

print(x_train.head())
print('')
print(x_train.shape)

#sns.lmplot(y_train,x_train['item_price'], fit_reg=False)
#sns.lmplot(x='item_cnt_day',y='item_price', data = df_train)
#plt.show()

lin_model = LinearRegression()
lin_model_fit = lin_model.fit(x_train,y_train)
lin_model_pred = lin_model_fit.predict(x_test)