import os

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
import pandas as pd

# 加载波士顿房价数据集
boston = load_boston()
X = boston.data
y = boston.target
data = pd.DataFrame(boston.data, columns=boston.feature_names)
price = pd.DataFrame(boston.target, columns=['Price'])

print(price['Price'].describe())
plt.figure(figsize=(9, 8))
plt.title('Price Distribution', fontsize=10)
sns.distplot(price['Price'], color='g', bins=100, hist_kws={'alpha': 0.4})
plt.savefig('result/Price_distribution.png')

data.hist(figsize=(24, 16), bins=50, xlabelsize=8, ylabelsize=8)
plt.suptitle('Histograms for Data Columns', fontsize=16)  # 添加总标题

# 保存到 result 文件夹中
plt.savefig(os.path.join('result/', 'histograms.png'))
# 计算特征之间的相关系数
correlation_matrix = data.corr()

# 使用Seaborn绘制热力图
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap of Boston Housing Dataset')
plt.savefig('result/relate.png')
plt.show()
