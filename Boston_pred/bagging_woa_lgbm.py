import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from woa import woa_algorithm

# 加载波士顿房价数据集
boston = load_boston()
X = boston.data
y = boston.target

# 将数据集划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 定义 LightGBM 参数优化的目标函数
def lgbm_objective_function(params):
    n_estimators, learning_rate, min_child_samples, subsample, max_depth = params

    # 初始化并训练 LightGBM 模型
    lgbm_model = LGBMRegressor(
        n_estimators=int(n_estimators),
        learning_rate=learning_rate,
        min_child_samples=int(min_child_samples),
        subsample=subsample,
        max_depth=int(max_depth),
        random_state=42
    )
    lgbm_model.fit(X_train, y_train)

    # 在测试集上进行预测
    y_pred = lgbm_model.predict(X_test)

    # 计算均方误差
    mse = mean_squared_error(y_test, y_pred)

    return mse

# 定义 LightGBM 参数优化的搜索空间
woa_bounds = np.array([(50, 200), (0.01, 0.1), (2, 20), (0.8, 1.0), (3, 10)])

# 定义 Bagging 集成学习模型参数
num_base_models = 5  # 设置基学习器个数 N
bagging_params = {
    'max_samples': 0.8,  # 在每轮选取数据集中多少比例的样本作为子训练集
    'max_features': 0.8,
}

# 初始化 BaggingRegressor 模型
bagging_model = BaggingRegressor(base_estimator=None, n_estimators=num_base_models, random_state=42, **bagging_params)

# Bagging 模型的最终预测结果
final_y_pred = np.zeros_like(y_test, dtype=float)

# Bagging 过程
for _ in range(num_base_models):
    # 使用 WOA 寻优 LightGBM 参数
    best_solution, _ = woa_algorithm(lgbm_objective_function, woa_bounds, population_size=10, max_iter=100)

    # 初始化并训练 LightGBM 模型
    lgbm_model = LGBMRegressor(
        n_estimators=int(best_solution[0]),
        learning_rate=best_solution[1],
        min_child_samples=int(best_solution[2]),
        subsample=best_solution[3],
        max_depth=int(best_solution[4]),
        random_state=42
    )
    lgbm_model.fit(X_train, y_train)

    # 在测试集上进行预测
    y_pred = lgbm_model.predict(X_test)

    # 累加到最终预测结果
    final_y_pred += y_pred

# 计算最终预测结果的均值
final_y_pred /= num_base_models

# 实际值和预测值的数据集
actual_values = y_test
predicted_values = final_y_pred

# 获取当前脚本的文件名（去掉 .py）
script_name = os.path.splitext(os.path.basename(__file__))[0]

# 计算均方根误差（RMSE）
rmse = np.sqrt(mean_squared_error(actual_values, predicted_values))

# 计算均方误差（MSE）
mse = mean_squared_error(actual_values, predicted_values)

# 计算平均绝对误差（MAE）
mae = mean_absolute_error(actual_values, predicted_values)

# 计算R方（R-squared）
r_squared = r2_score(actual_values, predicted_values)

# 创建包含结果的DataFrame
results_df = pd.DataFrame({
    'File': [script_name],
    'RMSE': [rmse],
    'MSE': [mse],
    'MAE': [mae],
    'R-squared': [r_squared]
})

# 打开CSV文件并追加数据
results_df.to_csv('analysis.csv', mode='a+', header=False, index=False)

# 保存图片到result文件夹中
result_folder = 'result'
os.makedirs(result_folder, exist_ok=True)

# 第一张图
plt.figure(figsize=(12, 6))
plt.scatter(range(len(y_test)), y_test, alpha=0.7, color='red', label='y_test')
plt.plot(range(len(final_y_pred)), final_y_pred, '--', color='blue', label='final_y_pred')
plt.title('Bagging-woa-lgbm:Scatter Plot of y_test vs. final_y_pred')
plt.xlabel('Sample Index')
plt.ylabel('Values')
plt.legend()
plt.savefig(os.path.join(result_folder, f"{script_name}_scatter_plot.png"))

# 第二张图
plt.figure(figsize=(12, 6))
residuals = y_test - final_y_pred
plt.scatter(range(len(final_y_pred)), residuals, alpha=0.7, color='blue')
plt.axhline(y=0, color='red', linestyle='--')
plt.title('Bagging-woa-lgbm:Residuals Plot')
plt.xlabel('Sample Index')
plt.ylabel('Residuals')
plt.savefig(os.path.join(result_folder, f"{script_name}_residuals_plot.png"))

plt.show()
