import random
from keras.models import Sequential
from keras.layers import Dense,Conv2D,Flatten
import numpy as np
import gym
import os
from collections import deque
import sys
import csv
from matplotlib import style
import matplotlib.pyplot as plt
import pickle
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

gamma = 0.9
TARGET_REPLACE_ITER = 25
nodes = 5
memory_capability = 500    #20000>1000
batch_size = 16    #25>30
epsilon_start =1
epsilon_end = 0.0001
epsilon_decay = 0.0015
target_update = 100
learning_rate = 0.0009
epsilon = 1
n_state = 2
n_action = 32        #50>26
sumnodes = np.zeros(nodes)
limit_decap = 1e-10
reward_list =[]
param_vmin = 0.95
vio_node = [0] * nodes
reward_max = []
reward_min = []
reward_avg = []
loss_val =[]


#RC随机参数
def rcval():
    os.system('./pdnrcval t1.conf t1.sp.rc')


#decap原始参数，全为0
def dcval_init():
    f = open('t1.sp.decap', 'w')
    for i in range(0, 5):

        f.write('.param cap_0_%d_val=0\n' % i)
    f.close()

decap_data = []
stringlist = []
decapsum = []
decap_val = []
for n in range(2):
    decap_data.append(n * 50e-12)    # 25*2

for a in decap_data:
    for b in decap_data:
        for c in decap_data:
            for d in decap_data:
                for e in decap_data:
                    string = ".param cap_0_0_val=%e\n.param cap_0_1_val=%e\n.param cap_0_2_val=%e\n.param cap_0_3_val=%e\n.param cap_0_4_val=%e\n" % (e,d,c,b,a)
                    stringlist.append(string)
                    decap_val.append((a,b,c,d,e))   # (4,3,2,1,0) -> (a,b,c,d,e)
                    decapsum.append(a+b+c+d+e)



class NN(nn.Module):
    
    def __init__(self, ):
        super(NN,self).__init__()
        self.fc1 = nn.Linear(n_state, 32)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(32,64)
        self.out = nn.Linear(64, n_action)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        action_value = self.out(x)
        return action_value



def ReadResult(filename):
    array = np.genfromtxt(filename, skip_header=1, skip_footer=1)
    return array

#原始ztotal
def origin_z_total():
    dcval_init()
    os.system("ngspice -b t1-ngspice.sp -r t1.raw > /dev/null 2> /dev/null")
    os.system("./raw2csv \"v(z_total)\" t1.raw > t1.sp.prn")
    z = ReadResult("t1.sp.prn")
    z_array = z[len(z) - 1][2:-1]
    z_total = z[len(z) - 1][-1]

    return z_total 

# 读取每次优化之后的z_total
def read_z_total():
    #aaa = np.zeros(shape=5)
    os.system("ngspice -b t1-ngspice.sp -r t1.raw > /dev/null 2> /dev/null")
    #for p in range(nodes):
    os.system("./raw2csv \"v(z_total)\" t1.raw > t_ztotal.sp.prn" )
    z = ReadResult("t_ztotal.sp.prn" )
    z_array = max(z[:,2])
    #aaa[p] = z_array

    return z_array 


def z_total_data():
    #dcval_init()
    os.system("ngspice -b t1-ngspice.sp -r t1.raw > /dev/null 2> /dev/null")
    os.system("./raw2csv \"v(z_total)\" t1.raw > t1.sp.prn")
    z = ReadResult("t1.sp.prn")
    z_time = z[:,1]
    z_total = z[:,2]

    return z_time,z_total 

def read_nvdd():
    aaa = np.zeros(shape=5)
    os.system("ngspice -b t1-ngspice.sp -r t1.raw > /dev/null 2> /dev/null")
    for p in range(nodes):
        os.system("./raw2csv \"v(nvdd_0_%d)\" t1.raw > t_%d.sp.prn" % (p,p))
        z = ReadResult("t_%d.sp.prn" % p)
        z_array = min(z[:,2])
        aaa[p] = z_array
    
    return aaa




class Pdn_Opt(object):

    def __init__(self,):
        # self.memsize = max_mem_size
        # self.state_size = state_size
        # self.action_size = action_size
        # self.model = self.creatmodel()
        #   # self.target_model = self.creatmodel()
        #         # self.memory = deque(maxlen=2000)
        self.learn_step_counter = 0
        self.memory = np.zeros((memory_capability, n_state * 2 + 2))
        self.memory_cntr = 0
        self.eval_net, self.target_net = NN(), NN()
        self.loss_func = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.eval_net.parameters(), lr=learning_rate)
        self.act_mode = 1
        self.Cof = 0
        self.ran_dc = 0
        self.ran_node = 0
        self.init_z = origin_z_total()
        self.state0 = (self.init_z * 1e10,0,0) 
        self.decap_param = [0,0,0,0,0]

    # def creatmodel(self):
    #     model = Sequential()
    #     model.add(Dense(24,input_dim= self.state_size,activation = 'relu'))
    #     model.add(Dense(24,activation='relu'))
    #     model.add(Flatten())
    #     model.add()
    #     return model


    
    def choose_action(self, state):
       # z_total = state[0]
        state = torch.unsqueeze(torch.FloatTensor(state),0)
        if random.random() < epsilon:
           # self.ran_dc = random.choice(decap_data)
          #  self.ran_node = random.randint(0, nodes - 1)
            action = random.randint(0,len(stringlist) - 1)
            self.act_mode = 1
        else:
            action_value = self.eval_net.forward(state)
            action = torch.max(action_value, 1)[1].numpy()[0]
            self.act_mode = 0
        return action

    def reset(self,):
        obs = np.zeros(shape=5)
        self.Cof = 0
        dcval_init()
        obs = read_z_total()
       # obs = Tensorfloat(obs)
        #for i in range(nodes):
            #vio_node[i] = 1 if obs[i] <= param_vmin else 0

        #total = (obs,self.Cof,vio_node)
        return (obs,self.Cof)

    
    def step(self,action):
       # z_total = state[0]
       # self.ran_node = random.choice([a for a,x in enumerate(vio_node) if x == 1])
       # decap1 = self.decap_param
        #str1 = '.param cap_0_%d_val=%e\n' % (self.ran_node, decap_data[action])
       # self.decap_param[self.ran_node] = decap_data[action]
        #decap2 =self.decap_param
       # while decap1 is decap2:
           # self.ran_node =random.randint(0,nodes - 1)
          #  self.decap_param[self.ran_node] = decap_data[action]
       # data = []                                   #创建一个列表，用于重新写入decap参数，全为字符串
        #for line in open('t1.sp.decap', 'r'):
        #    data.append(line)
        #data[self.ran_node] = str1
        f = open('t1.sp.decap', 'w')
        f.write(stringlist[action])
        #for i in range(nodes):
            #f.write(data[i])
        f.close()
        z_next = read_z_total()             #读取action之后的z_total
        #for abc in range(nodes):
           # if z_next[abc] <= param_vmin:
              #  vio_node[abc] = 1
           # else:
              #  vio_node[abc] = 0

        self.Cof = 1 if decapsum[action] > 1.005 * limit_decap else 0          #读取是否溢出

        obs = (z_next,self.Cof)

     #   if z_next <= z_total and self.Cof == 0 :
       #     reward = 10
      #  elif z_next <= z_total and self.Cof == 1:
      #      reward = -1
       # else:
      #      reward = -10

        return  obs


    def learn(self):
        # target net 参数更新
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # 抽取记忆库中的批数据
        sample_index = np.random.choice(memory_capability, batch_size)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :n_state])
        b_a = torch.LongTensor(b_memory[:, n_state:n_state + 1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, n_state + 1:n_state + 2])
        b_s_ = torch.FloatTensor(b_memory[:, -n_state:])
        b_r = torch.squeeze(b_r,1)
        # 针对做过的动作b_a, 来选 q_eval 的值, (q_eval 原本有所有动作的值)
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()  # q_next 不进行反向传递误差, 所以 detach
        q_target = b_r + gamma * q_next.max(1)[0]  # shape (batch, 1)
        q_target =torch.unsqueeze(q_target,1)
        loss = self.loss_func(q_eval, q_target)
        loss_val.append(loss)
        # 计算, 更新 eval net
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    # def save_position(self,state,action,reward,state_):
    #     index = self.mem_cntr % self.memsize
    #     self.memory.append((state,action,reward,state_))
    #     self.state_mem[index] = state
    #     self.action_mem[index] = action
    #     self.newstate_mem[index] = state_
    #     self.reward_mem[index] = reward
    #
    #     self.mem_cntr += 1


    def store_transition(self,state,action,reward,state_):
        transition = np.hstack((state,action,reward,state_))
        index = self.memory_cntr % memory_capability
        self.memory[index,:] = transition
        self.memory_cntr += 1


    #def reset(self):
        
       # return self.state0


   # def replay(self,batch_size):
     #   minibatch = random.sample(self.men)
ep = []
argtimes =[]
eps_list = []
pdn = Pdn_Opt()
init_z = origin_z_total()
z_time,init_ztotal = z_total_data()
#dcval_init()
episodes = 40
steps = 30
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
target_weight_list= []
eval_weight_list =[]

for i in range(episodes):
    dcval_init()  # every episodes reset 
    tar_data1 = pdn.target_net.out.weight.data
    eval_data = pdn.eval_net.out.weight.data
    if (i+1) % 5 ==0 or i ==0 :
        target_weight_list.append(tar_data1)
        eval_weight_list.append(eval_data)

    ep.append(i)
    argtime =0
    #rcval()
    init_z = origin_z_total()
    epi_reward = 0
    # state = (init_z,0,0)
    print('=' * 20)
    print('episode : %d' % i)
    min_z = init_z
    min_v = [0] *nodes
    vio_node = [0] * nodes
    state = pdn.reset()
    #state = min_z
   # vio_node = obs[2]
    eps_cnt = 0
    for mmm in range(steps):
        action = pdn.choose_action(state)        
        print('-'*10)
        #for h in range(nodes):
           # os.system("./raw2csv \"v(nvdd_0_%d)\" t1.raw > t_%d.sp.prn" % (h,h))
            #z = ReadResult('t_%d.sp.prn' % h)
            #min_v[h] = min(z[:,2])
           #vio_node[h] = 1 if min_v[h] < param_min else 0
           # obs[0][h] = min_v[h]
        eps_cnt += 1
       # state = obs[0]
       # action = pdn.choose_action(state)
        state_ = pdn.step(action)          # output 5nodes,cof,vio_node
        #reward = sum(-((state_ -param_vmin) * vio_node)**2 *500) - decapsum[action] * 0.2*1e11 + (nodes - sum(vio_node))if Cof == 0 else -1000
               # sum(-((state_ - param_vmin) * vio_node)**2 *5000) - sum(pdn.decap_param) * 0.2*1e11
        #  reward ^2
        reward = 1 -state_[0] / init_z if state_[1] == 0 else - state_[0] / init_z  - sum(decap_val[action]) / 2.5e-10
        
        if mmm == 0:
            maxreward = reward
            minreward = reward

        if maxreward < reward:
            maxreward = reward

        if minreward > reward:
            minreward = reward

       # if min_z >= state_[0] and Cof == 0:
          
        #    min_z = state_[0]
         #   reward = (init_z - min_z) *10
       # elif min_z < state_[0] and state_[1] == 0:
         #   reward = -(state_[0] - min_z)
        #else:
          #  reward =-10*math.fabs(init_z - state_[0])
        
       # if state_[0] <= min_z :
          #  min_z = state_[0]
        #if reward>= -15 and Cof == 0:
            #reward += 50
        pdn.store_transition(state,action,reward,state_)
        if pdn.act_mode == 1:
            print('take random action:%d\t ep: %d \n'%(action,i), 'z:{}\t z_:{}\n'.format(state,state_),'decap:\t',decap_val[action],'reward:',reward)
        elif pdn.act_mode == 0:
            print('take argmax action:%d\t ep:%d \n'%(action,i), 'z:{}\t z_:{}\n'.format(state,state_),'decap:\t',decap_val[action],'reward:',reward)
            argtime += 1

        epi_reward += reward


        if pdn.memory_cntr > memory_capability:
            pdn.learn() # 记忆库满了就进行学习

        
      #  if reward > 0  or epi_reward < -1000:
        state = state_
        #if reward >= -15 and Cof ==0:
           # reward += 50
           # break
    argtimes.append(argtime)
    reward_list.append(epi_reward)
    reward_avg.append(epi_reward / eps_cnt)
    reward_max.append(maxreward)
    reward_min.append(minreward)
    print('current :',state,'total reward=',epi_reward)
           # break
       # state = state_
    epsilon = epsilon_end +(epsilon_start - epsilon_end) * math.exp(-1. * 4/episodes * i)
    #epsilon = epsilon - 1./(episodes +1)
    eps_list.append(epsilon)
    #plt.plot(i,epi_reward)
   
print(type(tar_data1))
target1 = np.array(target_weight_list[0])
target2 =  np.array(target_weight_list[-1])

##eval1 = np.array(eval_weight_list)
np.savetxt('target_weight.txt',target1)
np.savetxt('target_weight2.txt',target2)

#np.savetxt('eval_weight.txt',eval1)

#tar_data1 = pdn.target_net.fc1.weight.data
#print(tar_data1)
Loss0 = torch.tensor(loss_val)
#print(loss_val)
z_time,end_ztotal = z_total_data()
aaa = read_nvdd()
nvdd_time = ReadResult('t_0.sp.prn')
z_a = []
for n in range(nodes):
    z =ReadResult('t_%d.sp.prn' % n)
    z_a.append(z[:,2])

reward_max_data = np.array(reward_max).reshape((int(episodes/5),5))
np.savetxt('rewardmax.txt',reward_max_data,'%.4f')
plt.subplot(2,2,1)
plt.plot(ep,reward_list)
plt.xlabel('epsiode')
plt.ylabel('reward')
plt.title('total reward')
#plt.figure()
plt.subplot(2,2,2)
plt.plot(ep,eps_list)
plt.xlabel('epsiode')
plt.ylabel('epsilon')
plt.title('exploration')
#plt.figure()
plt.subplot(2,2,3)
plt.plot(ep,argtimes)
plt.title('argmax action times')
plt.subplot(2,2,4)
plt.plot(ep,reward_max,label='max reward')
plt.plot(ep,reward_min,label='min reward')
plt.plot(ep,reward_avg,label='avg reward')
plt.xlabel('epsiodes')
plt.ylabel('reward')
plt.title('max/min/avg reward')
plt.legend()
plt.figure()
plt.plot([i for i in range(len(Loss0))],Loss0)
plt.xlabel('steps')
plt.ylabel('Loss value')
plt.title('Loss function')

plt.figure()
plt.subplot(2,3,1)
plt.plot(z_time,init_ztotal,label='before z_total')
plt.plot(z_time,end_ztotal,label='after z_total')
plt.title('before/after z_total' )
plt.xlabel('time')
plt.ylabel('voltage')
plt.legend()

plt.subplot(2,3,2)
plt.plot(nvdd_time,z_a[0])
plt.xlabel('time')
plt.ylabel('voltage ')
plt.title('Node 0 Voltage')
plt.subplot(2,3,3)
plt.plot(nvdd_time,z_a[1])
plt.xlabel('time')
plt.ylabel('voltage ')
plt.title('Node 1 Voltage')
plt.subplot(2,3,4)
plt.plot(nvdd_time,z_a[2])
plt.xlabel('time')
plt.ylabel('voltage')
plt.title('Node 2 Voltage')
plt.subplot(2,3,5)
plt.plot(nvdd_time,z_a[3])
plt.xlabel('time')
plt.ylabel('voltage ')
plt.title('Node 3 Voltage')
plt.subplot(2,3,6)
plt.plot(nvdd_time,z_a[4])
plt.xlabel('time')
plt.ylabel('voltage ')
plt.title('Node 4 Voltage')

plt.show()




#随机改变一个decap的参数
def action_choice():
    ran_dc = random.choice(decap_data)
    ran_node = random.randint(0, nodes - 1)
    str1 = '.param C_decap_%d=%e\n' % (ran_node,ran_dc)
    data = []
    for line in open('dcval.txt','r'):
        data.append(line)
    data[ran_node] = str1
    f = open('dcval.txt','w')
    for i in range(nodes):
         f.write(data[i])
    f.close()









#line = linecach
# rcval()
# dcval()
#
# os.system("ngspice -b decap_10.sp -r decap_10.raw > /dev/null 2> /dev/null")
# os.system("./raw2csv \"v(z_total)\" decap_10.raw > decap_10.sp.prn")
