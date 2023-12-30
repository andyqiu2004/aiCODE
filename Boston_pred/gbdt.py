import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Load the Boston housing dataset
boston = load_boston()
X = boston.data
y = boston.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Gradient Boosting model with limited tree depth
gbdt = GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, loss='ls')
gbdt.fit(X_train, y_train)

# Get the predicted values from the GBDT model
gbdt_test_pred = gbdt.predict(X_test)

# Create new features using GBDT predictions as input for Ridge regression

# Calculate mean squared error
mse_ensemble = mean_squared_error(y_test, gbdt_test_pred)
print("Ensemble Model Mean Squared Error:", mse_ensemble)

# 实际值和预测值的数据集
actual_values = y_test
predicted_values = gbdt_test_pred
final_y_pred = gbdt_test_pred

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
plt.title('gdbt:Scatter Plot of y_test vs. final_y_pred')
plt.xlabel('Sample Index')
plt.ylabel('Values')
plt.legend()
plt.savefig(os.path.join(result_folder, f"{script_name}_scatter_plot.png"))

# 第二张图
plt.figure(figsize=(12, 6))
residuals = y_test - final_y_pred
plt.scatter(range(len(final_y_pred)), residuals, alpha=0.7, color='blue')
plt.axhline(y=0, color='red', linestyle='--')
plt.title('gdbt:Residuals Plot')
plt.xlabel('Sample Index')
plt.ylabel('Residuals')
plt.savefig(os.path.join(result_folder, f"{script_name}_residuals_plot.png"))

plt.show()
