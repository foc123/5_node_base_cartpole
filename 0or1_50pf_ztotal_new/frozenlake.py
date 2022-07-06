import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import gym
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import math
nodes = 5
N_STATES = 2
N_ACTIONS = 2 ** nodes
lr = 0.001
gamma = 0.9
epsilon = 0.8
batch_size = 32
memory_size = 500
TARGET_REPLACE_ITER =25
per_decap = []
stringlist =[]
decap_list = []
decap_limit = 1e-10
loss_val =[]
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
for n in range(2):
    per_decap.append(n * 5e-11)

for e in per_decap:
    for d in per_decap:
        for c in per_decap:
            for b in per_decap:
                for a in per_decap:
                    strings = '.param cap_0_0_val=%e\n.param cap_0_1_val=%e\n.param cap_0_2_val=%e\n.param cap_0_3_val=%e\n.param cap_0_4_val=%e' % (a,b,c,d,e)
                    stringlist.append(strings)
                    decap_list.append((a,b,c,d,e))

def dcval_init():
    f = open('t1.sp.decap','w')
    for i in range(nodes):
        f.write('.param cap_0_%d_val=0\n' % i )
    f.close()

def readfile(filename):
    array = np.genfromtxt(filename,skip_header=1,skip_footer=1)
    return array


def read_vdi():
    os.system('ngspice -b t1-ngspice.sp -r t1.raw > /dev/null 2> /dev/null')
    os.system("./raw2csv \"v(z_total)\" t1.raw > t_ztotal.sp.prn")
    z = readfile('t_ztotal.sp.prn')
    time = z[:,1]
    z_total = z[-1][2]
   
    
    return time,z_total


def read_nvdd():
    a = np.zeros(shape=nodes)
    os.system('ngspice -b t1-ngspice -r t1.raw > /dev/null 2> /dev/null')
    for b in range(nodes):
        os.system('./raw2csv \'v(nvdd_0_%d)\' t1.raw > t_%d.sp.prn' % (b,b))
        z = readfile('t_%d.sp.prn'% b )
        zmin = min(z[:,2])
        a[b] = zmin
    return a



class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 50)
        self.fc1.weight.data.normal_(0, 0.1)  # initialization
        self.out = nn.Linear(50, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)   # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value


class Pdn:

    def __init__(self,):
        self.eval_net, self.target_net = Net(),Net()        
        self.memory_size = memory_size
        self.mem_count = 0
        self.memory = np.zeros([memory_size,N_STATES * 2 + 2])
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=lr)
        self.loss_func = nn.MSELoss()
        self.learn_step_counter = 0
        self.state = None
        self.ztime = None
        self.action_mode = None

    def step(self,action):
        decaps = decap_list[action]
        done = 1 if sum(decaps) > 1.001*decap_limit else 0
        f1 = open('t1.sp.decap','w')
        f1.write(stringlist[action])
        f1.close()
        self.ztime,ztotal = read_vdi()
        ss = (ztotal,done)
        return ss

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        # input only one sample
        if np.random.uniform() < epsilon:   # greedy
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy() 
            # output tensor(action_value) column's max,[1] means indices,.numpy can backward
            action = action[0]  # return the argmax index
            self.action_mode = 1
        else:   # random
            action = np.random.randint(0, N_ACTIONS - 1)
            self.action_mode = 0
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.mem_count % memory_size
        self.memory[index, :] = transition
        self.mem_count += 1

    def reset(self):
        dcval_init()  # decap
        vdi = read_vdi()[1]
        self.state =(vdi,0)
        return self.state

    def learn(self):
        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        sample_index = np.random.choice(memory_size, batch_size)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :N_STATES])
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2])
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])

        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()     # detach from graph, don't backpropagate
        q_target = b_r + gamma * q_next.max(1)[0].view(batch_size, 1)   # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)
        loss_val.append(loss)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

pdn = Pdn()
rewards = []
init_vdi = pdn.reset()[0]
init_nvdd =read_nvdd()
maxrs =[]
minrs =[]
avgrs=[]
episodes = 100
target_weight = []
t = 0
for i1 in range(episodes):
    print('=' * 30)
    print('ep:',i1)
    s = pdn.reset()
    ep_r = 0
    max_r =0
    avg_r =0
    min_r =0
    cnt = 0
    ov = 0
    if i1 % 25 == 0:
        target_weight.append((pdn.target_net.fc1.weight.data))
    #epsilon = math.exp(-4. / episodes * (i1+1))
    while True:
        t += 1
        a =pdn.choose_action(s)
        s_ = pdn.step(a)
        r = 1 - s_[0]/init_vdi   if s_[1] == 0 and s_[0] < 0.7* init_vdi else -1. * s_[0]/init_vdi
        pdn.store_transition(s,a,r,s_)
        ep_r += r
        cnt += 1
        print(s,a,r,s_,pdn.action_mode)
        if max_r < r:
            max_r = r
        if min_r > r:
            min_r = r
        if s_[1] == 1:
            ov += 1
        if pdn.mem_count > memory_size:
            pdn.learn()
            #if ov> 1 or ep_r >= 20:
            if i1 >= episodes / 2. and (cnt > 30 or ep_r > 20):
                print('eps:',i1,
                    'total rewards:',ep_r,
                      'counter:',cnt,
                      'optimized vdi:',s[0])

        if i1 < episodes /2. and ov > 3 or cnt > 40:
            break
        elif i1 >= episodes / 2. and ( ep_r >20 or cnt > 30):           
            break
        s = s_
    rewards.append(ep_r)

target1 = np.array(target_weight[0])
target2 = np.array(target_weight[-1])
target3 = np.array(target_weight[-1] - target_weight[0])
np.savetxt('out1_w.txt',target1)
np.savetxt('out200_w.txt',target2)
np.savetxt('out3_w.txt',target3)
Loss0 =torch.tensor(loss_val)

s1 = pdn.reset()
init_nvdd =read_nvdd()
s1 = torch.unsqueeze(torch.FloatTensor(s1), 0)
actions_value = pdn.eval_net.forward(s1)
action = torch.max(actions_value, 1)[1].data.numpy() 
a = action[0]
s1_ = pdn.step(a)
end_nvdd =read_nvdd()
print('action_value:\n',actions_value)
print('action:',action)
print('s,a,s_:',s1,a,s1_)

print(len(target_weight))
print("run times:",t)
plt.plot([x for x in range(episodes)],rewards)
plt.figure()
plt.plot([i for i in range(len(Loss0))],Loss0)
plt.xlabel('steps')
plt.ylabel('Loss value')
plt.title('Loss function')
plt.figure()
plt.scatter([b for b in range(5)],init_nvdd,label='init nvdd')
plt.scatter([b for b in range(5)],end_nvdd,label='end nvdd')
plt.legend()
plt.show()
