import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor

# 加载波士顿房价数据集
data = load_boston()
X = data.data
y = data.target

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 定义模型列表和结果字典
models = [KNeighborsRegressor(), RandomForestRegressor(), SVR(), GradientBoostingRegressor()]
results = {}

# 迭代50次求均值
n_iterations = 50
for model in models:
    mse_list = []
    for _ in range(n_iterations):
        # 训练模型
        reg_model = model.fit(X_train, y_train)
        # 在测试集上进行预测
        y_pred = reg_model.predict(X_test)
        # 计算均方根误差
        mse = mean_squared_error(y_test, y_pred)
        mse_list.append(mse)
    # 计算均值
    mean_mse = np.mean(mse_list)
    # 保存结果
    model_name = model.__class__.__name__
    results[model_name] = mean_mse

# 根据均方根误差选择最优模型
best_model = min(results, key=results.get)
if best_model == "GradientBoostingRegressor":
    best_model = 3

print(models[best_model])
# 绘制拟合图像
reg_model = models[best_model]
reg_model.fit(X_train, y_train)
y_pred = reg_model.predict(X)

plt.scatter(y, y_pred, color='r')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('Fitted Values vs True Values (Best Model: {})'.format(best_model))
plt.show()

# 打印均方根误差
for model_name, mse in results.items():
    print("{} 的均方根误差为: {}".format(model_name, mse))