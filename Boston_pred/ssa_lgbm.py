import os

import pandas as pd
from lightgbm import LGBMRegressor
from matplotlib import pyplot as plt
from skimage.metrics import mean_squared_error
from sklearn.datasets import load_boston
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import numpy as np

# best_solution = [1.21012692e+02, 6.42294081e-02, 6.46326886e+00, 9.70022283e-01, 6.52420630e+00] iter: 100/ 10
best_solution = [1.54328998e+02, 5.99737635e-02, 6.98909370e+00, 9.73205547e-01, 7.58864980e+00]  # iter: 100/100
# best_solution =
best_n_estimators, best_learning_rate, best_min_child_samples, best_subsample, best_max_depth = best_solution

final_lgbm_model = LGBMRegressor(
    n_estimators=int(best_n_estimators),
    learning_rate=best_learning_rate,
    min_child_samples=int(best_min_child_samples),
    subsample=best_subsample,
    max_depth=int(best_max_depth),
    random_state=42
)

# Load the Boston housing dataset
boston = load_boston()
X = boston.data
y = boston.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

final_lgbm_model.fit(X_train, y_train)
y_pred = final_lgbm_model.predict(X_test)

# 将y_pred转换为一维数组
y_pred = np.ravel(y_pred)

final_y_pred = y_pred
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

fontsize = 10.5 * 100 / 48
# 第一张图
plt.figure(figsize=(12, 6))
plt.scatter(range(len(y_test)), y_test, alpha=0.7, color='red', label='真实值')
plt.plot(range(len(final_y_pred)), final_y_pred, '--', color='blue', label='预测值')
plt.xlabel('样本号', fontsize=fontsize, fontname='SimSun')
plt.ylabel('价格/千美金', fontsize=fontsize, fontname='SimSun')
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.ylim([0, 60])
# 设置图例字体为宋体
font_prop = {'family': 'SimSun', 'size': fontsize}
plt.legend(prop=font_prop, loc='upper left')
plt.subplots_adjust(bottom=0.25)
plt.savefig(os.path.join(result_folder, f"{script_name}_scatter_plot.png"))
plt.show()

# 第二张图
plt.figure(figsize=(12, 6))
residuals = y_test - final_y_pred
plt.scatter(range(len(final_y_pred)), residuals, alpha=0.7, color='blue')
plt.axhline(y=0, color='red', linestyle='--')
# plt.title(f'{script_name}Residuals Plot')
plt.xlabel('样本号', fontsize=fontsize, fontname='SimSun')
plt.ylabel('残差/千美金', fontsize=fontsize, fontname='SimSun')
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.ylim([-10, 10])
plt.subplots_adjust(bottom=0.25)
plt.savefig(os.path.join(result_folder, f"{script_name}_residuals_plot.png"))

plt.show()