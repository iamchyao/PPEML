import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

data = pd.read_csv("potential_time.csv")


corr = data.corr()
plt.figure(figsize=(5,5),dpi=500)
plt.rc('font',family='Arial')
sns.heatmap(corr, linewidths=0.9, square=True, annot=True,cmap=sns.cubehelix_palette(as_cmap=True),vmin=-1.0,vmax=1.0)
plt.show()

feature = data.iloc[:,0:-2]
target_2 = data['FE%']

correlated_matrix = feature.corr()

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(feature,target_2,test_size=0.2,random_state=644)
x_test,x_val,y_test,y_val = train_test_split(x_test,y_test,test_size=0.5,random_state=644)
print(x_train.shape,y_train.shape)

from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

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
RFR_pred = RFR_model.predict(x_val)
mae = mean_absolute_error(y_val,RFR_pred)
print("val MAE = {:.4f}".format(mae))

mse = mean_squared_error(y_val,RFR_pred)
print("val MSE = {:.4f}".format(mse))

R2 = r2_score(y_val,RFR_pred)
print("val R2 = {:.4f}".format(R2))

print("val RMSE = {:.4f}".format(np.sqrt(mse)))

from xgboost import XGBRFRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedKFold

xgbr = XGBRFRegressor(random_state = 42)
xgbr_model = xgbr.fit(x_train,y_train)


xgbr_grid = {
    
    'min_child_weight': [1, 5, 10],
    'n_estimators': [4, 6, 8, 10, 12],
    'max_depth': [1,2, 3, 5],
    'learning_rate': [0.3, 1, 2],
    'colsample_bytree': [0.1, 0.5, 0.8, 1],
    'colsample_bylevel': [0.1, 0.5, 0.8, 1],

}
fold = RepeatedKFold(n_splits = 10,n_repeats = 1,random_state = 42)
xgbr_grid_cv = GridSearchCV(estimator = xgbr, cv = fold,
            param_grid = xgbr_grid, n_jobs = -1, scoring='neg_root_mean_squared_error',
            verbose = 1, return_train_score = True)

model = xgbr_grid_cv.fit(x_train,y_train,verbose = 1)
xgbr_preds = model.predict(x_train)

from sklearn.model_selection import cross_val_score

xgbr_optimized = xgbr_grid_cv.best_estimator_

print(xgbr_grid_cv.best_params_)

rmse_score = cross_val_score(xgbr_optimized, data, target_2, cv = fold, scoring='neg_root_mean_squared_error')
mae_score = cross_val_score(xgbr_optimized, data, target_2, cv = fold, scoring='neg_mean_absolute_error')
optimized_xgbr_rmse = np.abs(rmse_score).mean()
optimized_xgbr_mae = np.abs(mae_score).mean()


xgbr_preds_train = xgbr_model.predict(x_train)
xgbr_preds_1 = xgbr_model.predict(x_test)
xgbr_val_pred = xgbr_model.predict(x_val)

mae_train = mean_absolute_error(xgbr_preds_train,y_train)
mse_train = mean_squared_error(y_train,xgbr_preds_train)
R2_train = r2_score(y_train,xgbr_preds_train)
print('training R2 = {:.3f}'.format(R2_train))
print('training mae = {:.3f}'.format(mae_train))
print('training mse = {:.3f}'.format(mse_train))
print('training RMSE = {:.3f}'.format(np.sqrt(mse_train)))

print()

mae_test = mean_absolute_error(xgbr_preds_1 ,y_test)
mse_test = mean_squared_error(xgbr_preds_1 ,y_test)
R2_test = r2_score(y_test,xgbr_preds_1)
print('test R2 = {:.3f}'.format(R2_test))
print('test mae = {:.3f}'.format(mae_test))
print('test mse = {:.3f}'.format(mse_test))
print('test RMSE = {:.3f}'.format(np.sqrt(mse_test)))

print()
mae_val = mean_absolute_error(xgbr_val_pred,y_val)
mse_val = mean_squared_error(xgbr_val_pred,y_val)
R2_val = r2_score(y_val,xgbr_val_pred)
print('val R2 = {:.3f}'.format(R2_val))
print('val mae = {:.3f}'.format(mae_val))
print('val mse = {:.3f}'.format(mse_val))
print('val RMSE = {:.3f}'.format(np.sqrt(mse_val)))

from sklearn.ensemble import GradientBoostingRegressor

GBR = GradientBoostingRegressor()
GBR_model = GBR.fit(x_train,y_train)

parameters= {
    'n_estimators':list(range(1,30,2)),
    'max_depth':list(range(1,10,2)),

}

grid = GridSearchCV(estimator=GradientBoostingRegressor(random_state=2) ,param_grid=parameters,n_jobs= -1,cv=5)

GBR_model = grid.fit(x_train,y_train)
best_parameters = GBR_model.best_params_
print(best_parameters)

GBR_model = GBR_model.best_estimator_

y_GBR_pre_train = GBR_model.predict(x_train)
y_GBR_pred = GBR_model.predict(x_test)
y_GBR_val_pred = GBR_model.predict(x_val)


mae_train = mean_absolute_error(y_GBR_pre_train,y_train)
mse_train = mean_squared_error(y_GBR_pre_train,y_train)
R2_train = r2_score(y_train,y_GBR_pre_train)
print('training R2 = {:.3f}'.format(R2_train))
print('training mae = {:.3f}'.format(mae_train))
print('training mse = {:.3f}'.format(mse_train))
print('training RMSE = {:.3f}'.format(np.sqrt(mse_train)))

print()

mae_test = mean_absolute_error(y_GBR_pred,y_test)
mse_test = mean_squared_error(y_GBR_pred,y_test)
R2_test = r2_score(y_test,y_GBR_pred)
print('test R2 = {:.3f}'.format(R2_test))
print('test mae = {:.3f}'.format(mae_test))
print('test mse = {:.3f}'.format(mse_test))
print('test RMSE = {:.3f}'.format(np.sqrt(mse_test)))

print()
mae_val = mean_absolute_error(y_GBR_val_pred,y_val)
mse_val = mean_squared_error(y_GBR_val_pred,y_val)
R2_val = r2_score(y_val,y_GBR_val_pred)
print('val R2 = {:.3f}'.format(R2_val))
print('val mae = {:.3f}'.format(mae_val))
print('val mse = {:.3f}'.format(mse_val))
print('val RMSE = {:.3f}'.format(np.sqrt(mse_val)))

from sklearn.ensemble import AdaBoostRegressor

ada = AdaBoostRegressor()
ada_model = ada.fit(x_train,y_train)

parameters= {
    'n_estimators':list(range(1,30,2)),
}

grid = GridSearchCV(estimator=AdaBoostRegressor(random_state=2) ,param_grid=parameters,n_jobs= -1,cv=5)
ada_model = grid.fit(x_train,y_train)
best_parameters = grid.best_params_
print(best_parameters)


y_ada_pre_train = ada_model.predict(x_train)
y_ada_pred = ada_model.predict(x_test)
y_ada_val_pred = ada_model.predict(x_val)


mae_train = mean_absolute_error(y_ada_pre_train,y_train)
mse_train = mean_squared_error(y_ada_pre_train,y_train)
R2_train = r2_score(y_train,y_ada_pre_train)
print('training R2 = {:.3f}'.format(R2_train))
print('training mae = {:.3f}'.format(mae_train))
print('training mse = {:.3f}'.format(mse_train))
print('training RMSE = {:.3f}'.format(np.sqrt(mse_train)))

print()

mae_test = mean_absolute_error(y_ada_pred,y_test)
mse_test = mean_squared_error(y_ada_pred,y_test)
R2_test = r2_score(y_test,y_ada_pred)
print('test R2 = {:.3f}'.format(R2_test))
print('test mae = {:.3f}'.format(mae_test))
print('test mse = {:.3f}'.format(mse_test))
print('test RMSE = {:.3f}'.format(np.sqrt(mse_test)))

print()
mae_val = mean_absolute_error(y_ada_val_pred,y_val)
mse_val = mean_squared_error(y_ada_val_pred,y_val)
R2_val = r2_score(y_val,y_ada_val_pred)
print('val R2 = {:.3f}'.format(R2_val))
print('val mae = {:.3f}'.format(mae_val))
print('val mse = {:.3f}'.format(mse_val))
print('val RMSE = {:.3f}'.format(np.sqrt(mse_val)))