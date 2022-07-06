# 5_node_base_cartpole
五个节点的简单代码解释
训练样本为iload不变，R和C不变的情况下的电路模型，训练时每次重置均为不放置decap，将此时的VDI作为state输入agent训练。而测试集也为该电路，即测试电路和训练电路为同一个，因此不具有代表性，仅代表训练时可以输出最优解
reward定义为如果优化后的VDI在不超过decap放置限制的情况下，如果优化后VDI低于优化前的0.7倍，则reward为正数，否则为负数，即如下所示
![1657111271198](https://user-images.githubusercontent.com/89006608/177552420-1d6c0a4a-2ea1-4cfe-8056-6cc5d2c51e97.png)

训练后的loss function曲线

![1657115037389](https://user-images.githubusercontent.com/89006608/177566140-ebecdc01-18e5-44c5-91e5-222a41720ea9.png)

训练过程中的reward曲线，横坐标为episode index，纵坐标为每个episode的总reward

![1657115607589](https://user-images.githubusercontent.com/89006608/177566626-c54f4c2f-e710-46ec-944e-7b409bc546d0.png)

测试集的优化结果为如下

![1657116545006](https://user-images.githubusercontent.com/89006608/177570119-7071a098-9a02-47a3-886a-e833254bb64f.png)

横轴为节点坐标，纵轴为节点最低电压，当节点电压低于0.95v时算作发生了violate，纳入VDI计算

以下为部分超参数：
nodes = 5

N_STATES = 2

N_ACTIONS = 2 ^ nodes

lr = 0.001

gamma = 0.9

epsilon = 0.8

batch_size = 32

memory_size = 500

TARGET_REPLACE_ITER =25

decap_limit = 1e-10
