import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

data = pd.read_csv("potential_time.csv")

corr = data.corr()
plt.figure(figsize=(5,5),dpi=500)
sns.heatmap(corr, linewidths=0.9, square=True, annot=True,cmap=sns.cubehelix_palette(as_cmap=True))
plt.show()

feature = data.iloc[:,0:-2]
target_1 = data['Concentration']

correlated_matrix = feature.corr()

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(feature,target_1,test_size=0.2,random_state=580)
x_test,x_val,y_test,y_val = train_test_split(x_test,y_test,test_size=0.5,random_state=580)
print(x_train.shape,y_train.shape)

from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.metrics import r2_score


from sklearn.ensemble import RandomForestRegressor

RFR = RandomForestRegressor(random_state=2)
RFR_model = RFR.fit(x_train,y_train)

RFR_pred_train = RFR_model.predict(x_train)
mae = mean_absolute_error(y_train,RFR_pred_train)
print("train MAE = {:.4f}".format(mae))

mse = mean_squared_error(y_train,RFR_pred_train)
print("train MSE = {:.4f}".format(mse))

R2 = r2_score(y_train,RFR_pred_train)
print("train R2 = {:.4f}".format(R2))

print("train RMSE = {:.4f}".format(np.sqrt(mse)))
print()
RFR_pred = RFR_model.predict(x_test)
mae = mean_absolute_error(y_test,RFR_pred)
print("test MAE = {:.4f}".format(mae))

mse = mean_squared_error(y_test,RFR_pred)
print("test MSE = {:.4f}".format(mse))

R2 = r2_score(y_test,RFR_pred)
print("test R2 = {:.4f}".format(R2))

print("test RMSE = {:.4f}".format(np.sqrt(mse)))

print()
RFR_pred_val = RFR_model.predict(x_val)
print(RFR_pred_val)
mae = mean_absolute_error(y_val,RFR_pred_val)
print("val MAE = {:.4f}".format(mae))

mse = mean_squared_error(y_val,RFR_pred_val)
print("val MSE = {:.4f}".format(mse))

R2 = r2_score(y_val,RFR_pred_val)
print("val R2 = {:.4f}".format(R2))

print("val RMSE = {:.4f}".format(np.sqrt(mse)))

x_train_pred= RFR_model.predict(x_train)
x_train_pred = pd.DataFrame(x_train_pred)
data_train = pd.concat([y_train,x_train_pred],axis=1)


x = np.arange(0,4.1,0.1)
y = np.arange( 0,3.1,0.1)
X,Y = np.meshgrid(x,y)

X.tolist()

Y.tolist()

yy = []
for i in Y:
    yy = yy + list(i)


xx = []
for i in X:
    xx = xx + list(i)

data_val = pd.DataFrame({"cathode":xx,"anode":yy})

RFR_pred_data_val = RFR_model.predict(data_val)
data_pred_val = pd.DataFrame(RFR_pred_data_val)

data_rf = pd.concat([data_val,data_pred_val],axis=1)

data_rf.to_csv("grid_Concentration.csv",index=False)

data_train = pd.DataFrame(list(np.array(x_train.T))+[list(y_train),list(RFR_model.predict(x_train))]).T
data_val= pd.DataFrame(list(np.array(x_val.T))+[list(y_val),list(RFR_model.predict(x_val))]).T
data_test = pd.DataFrame(list(np.array(x_test.T))+[list(y_test),list(RFR_model.predict(x_test))]).T

data_train.to_csv("data_train_Concentration.csv")
data_val.to_csv("data_val_Concentration.csv")
data_test.to_csv("data_test_Concentration.csv")


