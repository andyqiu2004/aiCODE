import numpy as np


def woa_algorithm(objective_function, bounds, population_size, max_iter):
    dimension = len(bounds)

    # 初始化鲸鱼群体
    population = np.random.rand(population_size, dimension)
    population = bounds[:, 0] + population * (bounds[:, 1] - bounds[:, 0])

    # 初始化领头鲸的位置和适应值
    leader_position = np.zeros(dimension)
    leader_fitness = float('inf')

    # 主循环
    for iteration in range(max_iter):
        a = 2 - 2 * (iteration / max_iter)  # a 线性递减从 2 到 0

        for i in range(population_size):
            # 更新每条鲸鱼的位置
            r1 = np.random.rand(dimension)  # 随机向量
            r2 = np.random.rand(dimension)  # 随机向量

            A = 2 * a * r1 - a  # Equation (3.3)
            C = 2 * r2  # Equation (3.4)

            distance_to_leader = np.abs(leader_position - population[i, :])
            distance_to_leader = distance_to_leader * A  # Equation (2.8)

            new_position = leader_position - distance_to_leader
            new_position = np.clip(new_position, bounds[:, 0], bounds[:, 1])

            # 评估新位置的适应值
            fitness_new_position = objective_function(new_position)

            # 更新领头鲸
            if fitness_new_position < leader_fitness:
                leader_position = new_position
                leader_fitness = fitness_new_position

            # 更新当前鲸鱼的位置
            if fitness_new_position < objective_function(population[i, :]):
                population[i, :] = new_position

    return leader_position, leader_fitness


# 示例用法
def objective_function(x):
    # 这里替换成你的目标函数
    return np.sum(x ** 2)


# 定义搜索空间
bounds = np.array([[-5.12, 5.12], [-5.12, 5.12]])  # 两个维度的搜索范围

# 设置算法参数
population_size = 10
max_iter = 100

# 运行 WOA 算法
best_solution, best_fitness = woa_algorithm(objective_function, bounds, population_size, max_iter)

print("最优解:", best_solution)
print("最优解对应的目标函数值:", best_fitness)
