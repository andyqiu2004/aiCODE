import time

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
from ssa import ssa

# 加载波士顿房价数据集
boston = load_boston()
X = boston.data
y = boston.target

# 将数据集划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 起始时间
start_time = time.time()

# 定义用于LightGBM参数优化的目标函数
def lgbm_objective_function(params):
    n_estimators, learning_rate, min_child_samples, subsample, max_depth = params

    # 初始化并训练LightGBM模型
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

# 定义SSA搜索空间
dim = 5  # 要优化的参数数量
bounds = [(50, 200), (0.01, 0.1), (2, 20), (0.8, 1.0), (3, 10)]  # 每个参数的搜索范围

# 运行SSA优化LightGBM参数
best_solution, _ = ssa(lgbm_objective_function, dim, bounds, max_iter=100)

# 提取最优参数
best_n_estimators, best_learning_rate, best_min_child_samples, best_subsample, best_max_depth = best_solution

# 使用最优参数训练最终的LightGBM模型
final_lgbm_model = LGBMRegressor(
    n_estimators=int(best_n_estimators),
    learning_rate=best_learning_rate,
    min_child_samples=int(best_min_child_samples),
    subsample=best_subsample,
    max_depth=int(best_max_depth),
    random_state=42
)
final_lgbm_model.fit(X_train, y_train)

# 绘制优化过程（可选）
# 根据实际需求修改此部分
# 这只是一个基本示例，展示了优化过程的演变

# 初始化数组以存储每次迭代的最佳适应值
best_fitness_values = []

# 运行SSA算法并收集最佳适应值
best_solution, fitness_values = ssa(lgbm_objective_function, dim, bounds, max_iter=100)
best_fitness_values.append(fitness_values)

# 结束时间
end_time = time.time()
duration = end_time - start_time
print("ssa_LGBM算法的时间是", duration)

print("最优解:", best_solution)
print("最优解对应的目标函数值:", fitness_values)

