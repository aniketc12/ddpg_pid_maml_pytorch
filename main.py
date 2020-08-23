import sys
import gym
import matplotlib.pyplot as plt
from ddpg import DDPGagent
from utils import *
from pid_env import PidEnv

env = PidEnv(30)

agent = DDPGagent(env, 4, 256, 4)
noise = OUNoise(4)
batch_size = 64
rewards = []
avgRewards = []
metalearn = False
setpoints = []

if metalearn:
    for i in range(11):
        setpoints.append(np.random.random()*100)
    agent.metalearn(setpoints, setpoints)



for episode in range (800):
    state = env.reset()
    noise.reset()
    episodeReward = 0
    stepCounter = 0

    for step in range(400):
        stepCounter += 1
        action = agent.get_action(state)
        action = noise.get_action(action, step)
        new_state, reward, done = env.step(action)
        agent.memory.push(state, action, reward, new_state)

        if len(agent.memory) > batch_size:
            agent.update(batch_size)

        if episodeReward < -120000:
            print('Junk Episode')
            break

        if episode > 200:
            if episodeReward < -2000:
                break

        # kp = action[0]
        # ki = action[1]
        # kd = action[2]

        if done:
            sys.stdout.write(
                "episode: {}, reward: {}, average _reward: {} \n".format(episode, np.round(episodeReward, decimals=2),
                                                                         np.mean(rewards[-10:])))
            break

        state = new_state
        episodeReward += reward

    # print('Kp: ', kp, 'Ki: ', ki, 'Kd: ', kd)
    print('Episode: ', episode, ' StepCount', stepCounter)
    rewards.append(episodeReward)
    avgRewards.append(np.mean(rewards[-10:]))

# plt.plot(rewards)
plt.plot(avgRewards)
plt.plot()
plt.title('DDPG Model 1.0 - Training')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.show()
