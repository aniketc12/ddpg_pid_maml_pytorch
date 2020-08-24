import torch
import torch.autograd
import torch.optim as optim
import torch.nn as nn
from model import *
from utils import *
from pid_env import PidEnv


class DDPGagent:
    def __init__(self, env, numStates, numNNSize, numActions, actorLearningRate=1e-4, criticLearningRate=1e-3,
                 gamma=0.99, tau=0.99, maxMemSize=50000):
        # Params
        self.numStates = numStates
        self.numActions = numActions
        self.numNNSize = numNNSize
        self.gamma = gamma
        self.tau = tau

        self.agentUpdateLimCount = 0

        # Networks
        self.actor = Actor(self.numStates, self.numNNSize, self.numActions)
        self.actor_target = Actor(self.numStates, self.numNNSize, self.numActions)
        self.critic = Critic(self.numStates + self.numActions, self.numNNSize, self.numActions)
        self.criticTarget = Critic(self.numStates + self.numActions, self.numNNSize, self.numActions)

        for targetParam, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            targetParam.data.copy_(param.data)

        for targetParam, param in zip(self.criticTarget.parameters(), self.critic.parameters()):
            targetParam.data.copy_(param.data)

        # Training
        self.memory = Memory(maxMemSize)
        self.criticCriterion = nn.MSELoss()
        self.actorOptimizer = optim.Adam(self.actor.parameters(), lr=actorLearningRate)
        self.criticOptimizer = optim.Adam(self.critic.parameters(), lr=criticLearningRate)

    def get_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        action = self.actor.forward(state)
        action = action.data[0].tolist()
        return action

    def update(self, batchSize):
        self.agentUpdateLimCount += 1

        states, actions, rewards, nextStates = self.memory.sample(batchSize)
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards)
        nextStates = torch.FloatTensor(nextStates)

        # Critic loss
        Qvals = self.critic.forward(states, actions)
        nextActions = self.actor_target.forward(nextStates)
        nextQ = self.criticTarget.forward(nextStates, nextActions.detach())
        Qprime = rewards + self.gamma * nextQ
        criticLoss = self.criticCriterion(Qvals, Qprime)

        # Actor loss

        self.criticOptimizer.zero_grad()
        criticLoss.backward()
        self.criticOptimizer.step()

        policyLoss = -self.critic.forward(states, self.actor.forward(states)).mean()

        # update networks
        if self.agentUpdateLimCount % 2 == 0:

            self.actorOptimizer.zero_grad()
            policyLoss.backward()
            self.actorOptimizer.step()
            self.agentUpdateLimCount = 0


        # update target networks

        for targetParam, param in zip(self.criticTarget.parameters(), self.critic.parameters()):
            targetParam.data.copy_(param.data * self.tau + targetParam.data * (1.0 - self.tau))

        for targetParam, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            targetParam.data.copy_(param.data * self.tau + targetParam.data * (1.0 - self.tau))
            

    def metalearn(self, train_setpoints, test_setpoints):
        actorLearningRate=1e-4
        criticLearningRate=1e-3
        actor_theta = Actor(self.numStates, self.numNNSize, self.numActions)
        critic_theta = Critic(self.numStates + self.numActions, self.numNNSize, self.numActions)
        critic_theta_loss = nn.MSELoss()
        actor_theta_optimizer= optim.Adam(self.actor.parameters(), lr=actorLearningRate)
        critic_theta_optimizer = optim.Adam(self.critic.parameters(), lr=criticLearningRate)
        
        for outer in range(30):
            print(outer)
            outer_critic_loss = 0
            outer_actor_loss = 0
            for i in range(len(train_setpoints)):
                train_mem = Memory(10000)
                env = PidEnv(targetVal=train_setpoints[i])
                state = env.reset()
                eps_reward = 0
                inner_loss = 0
                #Before every trajectory, reinitialze theta parameters to original parameters
                for targetParam, param in zip(critic_theta.parameters(), self.critic.parameters()):
                    targetParam.data.copy_(param.data)

                for targetParam, param in zip(actor_theta.parameters(), self.actor.parameters()):
                    targetParam.data.copy_(param.data)

                for step in range(400):
                    state_push = state
                    state = torch.from_numpy(state).float().unsqueeze(0)
                    action = self.actor.forward(state)
                    action = action.data[0].tolist()
                    new_state, reward, done = env.step(action)
                    train_mem.push(state_push, action, reward, new_state)

                    #Inner loss is negative sum of expected reward which is known using the critic theta network
                    inner_loss -= critic_theta(state, actor_theta(state)).mean()

                    state = new_state

                    #break loop if error is too high or episode is completed
                    if done == True or eps_reward < -1000:
                        break

                #update critic theta's parameters using mean squared error loss
                size = int(step/2) if step > 10 else step
                states, actions, rewards, next_states = train_mem.sample(size)
                states = torch.FloatTensor(states)
                actions = torch.FloatTensor(actions)
                rewards = torch.FloatTensor(rewards)
                next_states = torch.FloatTensor(next_states)
                qvals = critic_theta(states, actions)
                next_actions = actor_theta(next_states)
                next_q = critic_theta(next_states, next_actions.detach())
                target_q = rewards + self.gamma * next_q
                critic_loss = critic_theta_loss(qvals, target_q)
                critic_theta_optimizer.zero_grad()
                critic_loss.backward()
                critic_theta_optimizer.step()

                #Update actor theta using inner loss(which is the negative sum of expected reward) as the loss function
                actor_theta_optimizer.zero_grad()
                actor_grads = torch.autograd.grad(inner_loss, actor_theta.parameters(), create_graph=True)
                for parameter, grad in zip(actor_theta.parameters(), actor_grads):
                    parameter.data.copy_(parameter - 0.005*grad)

                train_mem = Memory(10000)
                env = PidEnv(targetVal=train_setpoints[i])
                state = env.reset()
                eps_reward = 0
                done = False
                for step in range(400):
                    state_push = state
                    state = torch.FloatTensor(state).unsqueeze(0)
                    action = actor_theta(state).data[0].tolist()
                    new_state, reward, done = env.step(action)
                    train_mem.push(state_push, action, reward, new_state)

                    #Inner loss is negative sum of expected reward which is known using the critic theta network
                    outer_actor_loss -= critic_theta(state, actor_theta(state)).mean()

                    state = new_state
                    eps_reward += reward

                    #break loop if error is too high or episode is completed
                    if step > 2 and (done == True or eps_reward < -1000):
                        break

                #update critic theta's parameters using mean squared error loss
                size = int(step/2) if step > 10 else step
                states, actions, rewards, next_states = train_mem.sample(size)
                states = torch.FloatTensor(states)
                actions = torch.FloatTensor(actions)
                rewards = torch.FloatTensor(rewards)
                next_states = torch.FloatTensor(next_states)
                qvals = critic_theta(states, actions)
                next_actions = actor_theta(next_states)
                next_q = critic_theta(next_states, next_actions.detach())
                target_q = rewards + self.gamma * next_q
                outer_critic_loss += critic_theta_loss(qvals, target_q)

            self.actorOptimizer.zero_grad()
            outer_actor_loss.backward()
            self.actorOptimizer.step()

            self.criticOptimizer.zero_grad()
            outer_critic_loss.backward()
            self.criticOptimizer.step()

