# 5_node_base_cartpole
五个节点的简单代码解释
训练样本为iload不变，R和C不变的情况下的电路模型，训练时每次重置均为不放置decap，将此时的VDI作为state输入agent训练。而测试集也为该电路，因此不具有代表性，仅代表训练时可以输出最优解
reward定义为如果优化后的VDI在不超过decap放置限制的情况下，如果优化后VDI低于优化前的0.7倍，则reward为正数，否则为负数，即如下所示
![1657111271198](https://user-images.githubusercontent.com/89006608/177552420-1d6c0a4a-2ea1-4cfe-8056-6cc5d2c51e97.png)

训练后的loss function曲线

![1657115037389](https://user-images.githubusercontent.com/89006608/177566140-ebecdc01-18e5-44c5-91e5-222a41720ea9.png)
