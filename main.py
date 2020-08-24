import sys
import gym
import matplotlib.pyplot as plt
from ddpg import DDPGagent
from utils import *
from pid_env import PidEnv


env = PidEnv()
agent = DDPGagent(env, 4, 256, 4)
noise = OUNoise(4)
batch_size = 256
rewards = []
avgRewards = []
normalized = []
metalearn = False
random = False

for episode in range(300):

    sp = 50 if random == False else np.random.random()*100
    env = PidEnv(sp)
    state = env.reset()
    noise.reset()
    episodeReward = 0
    stepCounter = 0

    for step in range(250):
        stepCounter += 1
        action = agent.get_action(state)
        action = noise.get_action(action, step)
        new_state, reward, done = env.step(action)
        agent.memory.push(state, action, reward, new_state)

        if len(agent.memory) > batch_size:
            agent.update(batch_size)



        if episodeReward < -10000:
            print('Junk Episode')
            break

        state = new_state
        episodeReward += reward

        #   kp = action[0]
        #   ki = action[1]
        #   kd = action[2]

        if done:
            if reward > 0:
                print('Sucess')
                env.render()
                normalized.append(reward)
        #    print(step)
        #    sys.stdout.write(
        #        "episode: {}, reward: {}, average _reward: {} \n".format(episode, np.round(episodeReward, decimals=2),
        #                                                                 np.mean(rewards[-10:])))
            break


    # print('Kp: ', kp, 'Ki: ', ki, 'Kd: ', kd)
    print('Episode: ', episode, ' StepCount', stepCounter)
    rewards.append(episodeReward)
    avgRewards.append(np.mean(rewards[-10:]))

agent2 = DDPGagent(env, 4, 256, 4)
rewards2 = []
avgRewards2 = []
if metalearn == True:
    setpoints = []
    for i in range(20):
        sp = 50 if random == False else np.random.random()*100
        setpoints.append(sp)

    agent2.metalearn(setpoints, setpoints)

    for episode in range(300):

        sp = 50 if random == False else np.random.random()*100
        env = PidEnv(sp)
        state = env.reset()
        noise.reset()
        episodeReward = 0
        stepCounter = 0

        for step in range(250):
            stepCounter += 1
            action = agent2.get_action(state)
            action = noise.get_action(action, step)
            new_state, reward, done = env.step(action)
            agent2.memory.push(state, action, reward, new_state)

            if len(agent2.memory) > batch_size:
                agent2.update(batch_size)

            if episodeReward < -10000:
                print('Junk Episode')
                break

            state = new_state
            episodeReward += reward
            if done:
                if reward > 0:
                    print('Sucess')
                    env.render()
                break


        print('Episode: ', episode, ' StepCount', stepCounter)
        rewards2.append(episodeReward)
        avgRewards2.append(np.mean(rewards2[-10:]))



# plt.plot(rewards)
if metalearn == True:
    plt.plot(avgRewards2)
    plt.text(len(avgRewards2)-1, avgRewards2[-1], 'Rewards with Meta-Learning')
plt.plot(avgRewards)
plt.text(len(avgRewards)-1, avgRewards[-1], 'Rewards')
plt.title('DDPG Model 1.1 - Training')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.show()
