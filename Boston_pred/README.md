# 一些优越性验证

ssa.py 麻雀优化算法

woa.py 鲸鱼优化算法

ssa_gbdt.py  基于ssa的gdbt

ssa_lgbm.py  基于ssa的lgbm

bagging_ssa_lgbm.py  基于bagging-ssa的lgbm

bagging_woa_lgbm.py  基于bagging-woa的lgbm

运行上面两个文件可以看到每次优化出发点很快就相同，验证Bagging无法很好利用LightGBM

result文件中是相关的拟合曲线，以及对于数据集的热力图

other_methods中包含验证gbdt相比于其他四种模型的优越性，对于网格优化mse>9劣于普通的gdbt于是不加入讨论。
对于岭回归load_Boston_Ridge的mse>20同样不参与讨论。

