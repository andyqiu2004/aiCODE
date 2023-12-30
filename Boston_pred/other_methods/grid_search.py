# import numpy as np
# from sklearn.datasets import load_boston
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.ensemble import GradientBoostingRegressor
# from sklearn.metrics import mean_squared_error
#
# # 加载波士顿房价数据集
# boston = load_boston()
# X = boston.data
# y = boston.target
#
# # 划分训练集和测试集
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # 定义GBDT模型
# gbdt_model = GradientBoostingRegressor()
#
# # 定义参数网格
# param_grid = {
#     'n_estimators': [50, 100, 200],
#     'learning_rate': [0.01, 0.1, 0.2],
#     'max_depth': [3, 5, 7],
#     'subsample': [0.8, 1.0],
#     'max_features': [0.8, 1.0]
# }
#
# # 使用GridSearchCV进行参数搜索
# grid_search = GridSearchCV(estimator=gbdt_model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5)
# grid_search.fit(X_train, y_train)
#
# # 输出最优参数
# best_params = grid_search.best_params_
# print("最优参数:", best_params)
#
# # 使用最优参数的模型进行训练
# best_gbdt_model = GradientBoostingRegressor(**best_params)
# best_gbdt_model.fit(X_train, y_train)
#
# # 在测试集上进行预测
# y_pred = best_gbdt_model.predict(X_test)
#
# # 输出在测试集上的均方误差
# mse = mean_squared_error(y_test, y_pred)
# print("在测试集上的均方误差:", mse)


import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

# 加载波士顿房价数据集
boston = load_boston()
X = boston.data
y = boston.target

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 定义GBDT模型
gbdt_model = GradientBoostingRegressor()

# 定义参数网格
param_grid = {
    'n_estimators': [50, 100, 150, 200],
    'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
    'max_depth': [3, 4, 5, 6, 7],
    'subsample': [0.8, 0.9, 1.0],
    'max_features': [0.8, 0.9, 1.0]
}

# 使用GridSearchCV进行参数搜索
grid_search = GridSearchCV(estimator=gbdt_model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5)
grid_search.fit(X_train, y_train)

# 输出最优参数
best_params = grid_search.best_params_
print("最优参数:", best_params)

# 使用最优参数的模型进行训练
best_gbdt_model = GradientBoostingRegressor(**best_params)
best_gbdt_model.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = best_gbdt_model.predict(X_test)

# 输出在测试集上的均方误差
mse = mean_squared_error(y_test, y_pred)
print("在测试集上的均方误差:", mse)
