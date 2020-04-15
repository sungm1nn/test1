import math
import gym
import subprocess
from gym import spaces, logger
from gym.utils import seeding
import numpy as np

# PARA = "kernel.sched_min_granularity_ns"
# DEF_PARA = 3000000

class envTestEnv(gym.Env):
    def __init__(self):

        self.para1=list(range(200))
        self.flag1 = True
        self.cnt = 1
        for self.x in range(0, 200):
            if self.flag1:
                self.para1[self.x] = self.cnt
                self.cnt=self.cnt+1
            else :
                self.para1[self.x] = self.cnt
                self.cnt = self.cnt-1
            if self.cnt == 100:
                self.flag1 = False

        # self.para = PARA
        self.para_val = 150
        # self.ipc = 0
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Discrete(4)
        self.x = 150
        self.state = self.para1[self.para_val]

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        # self.para_val = self.x
        #self.state = self.para1[self.para_val]
        next_ipc = self.para1[self.para_val]
        if action == 0:
            self.para_val = self.para_val + 1
            # self.para_val = self.change_para(self.para_val, 500)
        else:  # semi-implicit euler
            # self.para_val = self.change_para(self.para_val, -500)
            self.para_val = self.para_val - 1

        #next_ipc = self.get_next_ipc()

        if self.para_val >= 199 or self.para_val<=0:
            reward =0
            done = True
        elif next_ipc < self.para1[self.para_val]:
            reward = 100
            self.next_state = self.para1[self.para_val]
            done = False
        elif next_ipc > self.para1[self.para_val]:
            reward = -100
            self.next_state = self.para1[self.para_val]
            done = False
        else:
            reward = 0
            done = True
        #print(reward)
        #ipc = self.para1[self.x]
        # if self.para1[self.para_val] >= 90 :
        #    print(str(self.para1[self.para_val]) + " yes yes good!")
        # next_state = self.para1[self.para_val]
        # if self.para1[]
        # print(self.ipc)

        return np.array(self.next_state), reward, done, {}

    def reset(self):
        self.x = 150
        self.para_val = self.x  # self.change_para(DEF_PARA, 0)

        self.state = self.para1[self.para_val]
        return np.array(self.state)


'''
    def get_next_ipc(self):
        #cmd = 'perf stat -a -o p.log ./hackbench.x64  &>/dev/null &'
        cmd = 'perf stat -a -o p.log sleep 1  &>/dev/null &'
        cmd1 = 'cat p.log |grep inst |awk \'{print $1}\'|tr -d ,'

        subprocess.run(cmd, stdout=subprocess.PIPE, shell=True)
        p2 = int(subprocess.check_output(cmd1, shell=True, encoding='utf-8'))
        return p2

    def change_para(self, val, next_val):
        self.para_val =  val + next_val
        tunecmd = "sysctl -w " + self.para + "=" + str(self.para_val)
        subprocess.run(tunecmd, stdout=subprocess.PIPE, shell=True)
        print("PARA_val = " + str(self.para_val))
        return self.para_val
'''