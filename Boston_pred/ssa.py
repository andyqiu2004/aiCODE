import numpy as np

def ssa(objective_function, dim, bounds, max_iter):
    """
    麻雀搜索算法

    参数:
    - objective_function: 目标函数，接受一个参数（解）并返回一个标量
    - dim: 解的维度
    - bounds: 解的边界，格式为 [(min1, max1), (min2, max2), ..., (min_dim, max_dim)]
    - max_iter: 迭代次数

    返回:
    - best_solution: 最优解
    - best_fitness: 最优解对应的目标函数值
    """

    def initialize_population(dim, bounds, size):
        """
        初始化种群
        """
        population = np.zeros((size, dim))
        for i in range(dim):
            population[:, i] = np.random.uniform(bounds[i][0], bounds[i][1], size)
        return population

    def get_fitness(population):
        """
        计算种群中每个个体的适应度（目标函数值）
        """
        return np.array([objective_function(ind) for ind in population])

    def move_towards_target(population, target, c1, c2):
        """
        根据当前位置和目标位置，更新种群中个体的位置
        """
        velocity = np.zeros_like(population)
        for i in range(dim):
            velocity[:, i] = velocity[:, i] + c1 * np.random.rand() * (target[i] - population[:, i])
            velocity[:, i] = velocity[:, i] + c2 * np.random.rand() * (global_best[i] - population[:, i])
            population[:, i] = population[:, i] + velocity[:, i]
        return population

    # 初始化种群
    population_size = 20
    population = initialize_population(dim, bounds, population_size)

    # 初始化全局最优解
    global_best = population[np.argmin(get_fitness(population))]
    global_best_fitness = np.min(get_fitness(population))

    # 设置算法参数
    c1 = 0.5  # 加速系数1
    c2 = 0.5  # 加速系数2

    # 开始迭代
    for iter in range(max_iter):
        # 计算适应度
        fitness_values = get_fitness(population)

        # 更新全局最优解
        if np.min(fitness_values) < global_best_fitness:
            global_best = population[np.argmin(fitness_values)]
            global_best_fitness = np.min(fitness_values)

        # 移动个体向全局最优解和当前最优解的方向
        population = move_towards_target(population, global_best, c1, c2)

    best_solution = global_best
    best_fitness = global_best_fitness

    return best_solution, best_fitness

# 示例目标函数，这里使用一个简单的二维函数
def objective_function(x):
    return x[0]**2 + x[1]**2

# # 定义解的维度和边界
# dim = 2
# bounds = [(-5, 5), (-5, 5)]
#
# # 运行麻雀搜索算法
# best_solution, best_fitness = ssa(objective_function, dim, bounds, max_iter=100)
#
# # 打印结果
# print("最优解:", best_solution)
# print("最优解对应的目标函数值:", best_fitness)
