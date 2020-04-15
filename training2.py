import gym
import numpy as np
import random
from keras.models     import Sequential
from keras.layers     import Dense
from keras.optimizers import Adam  
from gym.envs.registration import register
from matplotlib import pyplot as plt

register(
    id='EnvTest-v0',
    entry_point = 'gym.envTest_1:envTestEnv',
)

env = gym.make('EnvTest-v0')
# env = gym.make('CartPole-v1')
goal_steps = 500

'''
for i in range(200):
  obs = env.reset()
  for i in range(goal_steps):
    obs, reward, done, info = env.step(random.randrange(0, 2))
    if done: break
    # env.render()
'''

def data_preparation(N, K, f, render=False, v=False):
  game_data = []
  for i in range(N):
    score = 0
    game_steps = []
    obs = env.reset()
    for step in range(goal_steps):
      # if render: env.render()
      action = f(obs)
      game_steps.append((obs, action))
      obs, reward, done, info = env.step(action)
      score += reward
      if done or score <= -1000:
        break
    # print(score)
    game_data.append((score, game_steps))
  
  game_data.sort(key=lambda s:-s[0])
  # print(game_data)
  training_set = []
  # tmp_array = list(range(2), range(0, K))
  tmp = []
  f=open('C:/Users/Sungmin/Desktop/hi.txt', 'w')
  for i in range(K):
    for step in game_data[i][1]:
      if v==True:
         tmp.append(step[0])
      # print(step[0])
      if step[1] == 0:
        training_set.append((step[0], [1, 0]))
      else:
        training_set.append((step[0], [0, 1]))
    if v==True:
      f.write(str(game_data[i][1]))
      # for j in game_data[i][1] :
      #  f.write(str(step))
      # print(tmp[7:9])
      '''
      for p in range(0, len(tmp)):
        tmp[p] = tmp[7:9]
        print(tmp[p])
      '''
      inttmp=[]
      for a in tmp:
        if a != '':
          hello = str(a)
          inttmp.append(int(hello))
      plt.plot(inttmp, marker='.')
      plt.savefig('C:/Users/Sungmin/Desktop/rl/hi/savefig'+str(i)+".png")
      plt.clf()

      for p in range(0, len(tmp)):
        del tmp[0]
      for p in range(0,len(inttmp)):
        del inttmp[0]

    f.write("\n")
    f.write("\n")
    print(i, " @@@@@@@@@ ")


  print("{0}/{1}th score: {2}".format(K, N, game_data[K-1][0]))
  # if render:
  # print(game_data)
  for i in game_data:
    print("Score: {0}".format(i[0]))

  return training_set

def build_model():
  model = Sequential()
  model.add(Dense(128, input_dim=1, activation='relu'))
  model.add(Dense(52, activation='relu'))
  model.add(Dense(2, activation='softmax'))
  model.compile(loss='mse', optimizer=Adam())
  return model

def train_model(model, training_set):
  X = np.array([i[0] for i in training_set]).reshape(-1, 1)
  y = np.array([i[1] for i in training_set]).reshape(-1, 2)
  model.fit(X, y, epochs=10)

# def draw_graph(x_axis,y_axis):
  

if __name__ == '__main__':
  N = 500
  K = 50
  self_play_count = 10
  model = build_model()
  training_data = data_preparation(N, K, lambda s: random.randrange(0, 2))
  train_model(model, training_data)
  print("1st train is done")
  def predictor(s):
    return np.random.choice([0, 1], p=model.predict(s.reshape(-1, 1))[0])

  for i in range(self_play_count):
    K = (N//9 + K)//2
    training_data = data_preparation(N, K, predictor)
    train_model(model, training_data)
    print(str(i+1)+ " ahahahahahahaha")
  # print(model)
  data_preparation(100, 100, predictor, False, True)