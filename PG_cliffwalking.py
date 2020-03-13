import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from keras import backend as K
print(keras.__version__)

import sys
sys.path.append('../')
from env.cliff_walking import CliffWalkingEnv
import numpy as np
import matplotlib.pyplot as plt
from running_variance import RunningVariance

#random seed
np.random.seed(123)
n_episode = 1000


env = CliffWalkingEnv()
env.reset()


class CliffWalkingPG:
    def __init__(self, render=False):
        self.render = True

        self.state_dim = env.observation_space.n
        self.action_count = env.action_space.n
        self.update_frequency = 5
        self.discount_factor = 0.9
        self.running_variance = RunningVariance()

        print('state_dim: {}, action_count: {}, update_frequency: {} '.format(
        self.state_dim, self.action_count, self.update_frequency))

        #actor network
        actor = Sequential()
        actor.add(Dense(16, input_shape=(self.state_dim,), activation='relu', kernel_initializer='he_uniform'))
        actor.add(Dense(self.action_count, activation='softmax', kernel_initializer='he_uniform'))
        actor.summary()
        self.actor = actor

        #actor_optimizer
        action = K.placeholder(shape=[None, self.action_count])
        advantage = K.placeholder(shape=[None, ])

        action_prob = K.sum(action - self.actor.output, axis=1)
        cross_entropy = K.log(action_prob) * advantage
        loss = -K.sum(cross_entropy)

        optimizer = Adam(lr=0.005)
        updates = optimizer.get_updates(self.actor.trainable_weights, [], loss)
        train = K.function([self.actor.input, action, advantage], [], updates=updates)
        self.actor_optimizer = train

        #critic network
        critic = Sequential()
        critic.add(Dense(16, input_shape=(self.state_dim,), activation='relu', kernel_initializer='he_uniform'))
        critic.add(Dense(1, activation='linear', kernel_initializer='he_uniform'))
        critic.summary()
        self.critic = critic

        #critic optimizer
        target = K.placeholder(shape=[None, ])
        loss = K.mean(K.square(target - self.critic.output))

        optimizer = Adam(lr=0.001)
        updates = optimizer.get_updates(self.critic.trainable_weights, [], loss)
        train = K.function([self.critic.input, target], [], updates=updates)
        self.critic_optimizer = train

        #compile optimizer

    def get_action(self, state, e):
        if np.random.rand(1) > e:
            #need to put exploration-exploitation
            policy = self.actor.predict(state, steps=1).flatten() #2d -> 1d
            print(policy)
            #return np.random.choice(self.action_count, 1, p=policy)[0] #p : probability.
            return np.argmax(policy)
        else:
            return np.random.choice(self.action_count, 1)[0] #epsilon_greedy

    def train_model(self, states, actions, rewards, dones):
        states = np.array(states)
        states = np.squeeze(states, axis=1)

        #calculate discounted reward
        discounted_rewards = self.get_discount_reward(rewards, 0.9)

        #train critic network
        for r in range(10):
            self.critic_optimizer([states, discounted_rewards])

        #train actor network using baseline
        act = np.zeros((len(actions), self.action_count), dtype=np.float32)
        act[np.arange(len(actions)), actions] = 1

        baseline = self.critic.predict(states).flatten()
        advantage = discounted_rewards - baseline
        advantage = advantage
        for r in advantage:
            self.running_variance.add(r)
        self.actor_optimizer([states, act, advantage])

        out = self.actor.predict(states)
        action_prob = K.sum(act * out, axis=1)
        print('action, out, ...', action, out, act, K.eval(action_prob))
        cross_entropy = K.log(action_prob) * advantage
        loss = -K.sum(cross_entropy)
        print('loss : ', K.eval(loss), K.eval(action_prob), K.eval(K.log(action_prob) * advantage))
        #action prob가 0, 1, 2가 나올수 있는지 조사하기
    def get_discount_reward(self, reward, gamma=0.9):
        discounted_prediction = np.zeros_like(reward, dtype=np.float32)
        running_add = 0
        for i in reversed(range(0, len(reward))):
            running_add = running_add * gamma + reward[i]
            discounted_prediction[i] = running_add
        return discounted_prediction

def state_to_ary(state):
    state = np.ravel_multi_index(state, env.shape)
    ary = np.zeros((1, env.observation_space.n))
    ary[0, state] = 1
    return ary


def draw_result(episodes, scores, n_step):
    plt.subplot(121)
    plt.plot(episodes, scores, 'b')
    plt.xlabel("episodes")
    plt.ylabel("scores")
    plt.title("scores")

    plt.subplot(122)
    plt.plot(episodes, n_step, 'b')
    plt.xlabel("episodes")
    plt.ylabel("n_step")
    plt.title("n_step")
    plt.show()


class Observations:
    def __init__(self):
        self.reset()

    def reset(self):
        self.d = {'state':[], 'action':[], 'reward':[], 'done':[]}

    def add(self, state, action, reward, done):
        self.d['state'].append(state)
        self.d['action'].append(action)
        self.d['reward'].append(reward)
        self.d['done'].append(done)

    def get_observations(self):
        return self.d['state'], self.d['action'], self.d['reward'], self.d['done']


if __name__=='__main__':
    action_size = env.action_space.n
    agent = CliffWalkingPG()
    observations = Observations()
    scores, episodes, steps = [], [], []

    cnt = 0
    for e in range(n_episode):
        done = False
        score = 0
        state = env.reset()
        state = state_to_ary(state)
        cnt += 1

        observations.reset()
        print('episode : {}, Variance : {}'.format(e, agent.running_variance.get_variance()))
        n_step = 0
        while not done:
            n_step += 1
            if agent.render:
                env.render()

            action = agent.get_action(state, max(1 - 2*e/n_episode, 0))
            print('state, action : ', np.unravel_index(np.argmax(state[0]), env.shape), action)
            next_state, reward, done, info = env.step(action)
            next_state = state_to_ary(next_state)
            observations.add(state, action, reward, done)
            score += reward
            state = next_state

            if done:
                steps.append(n_step)
                states, actions, rewards, dones = observations.get_observations()
                agent.train_model(states, actions, rewards, dones)

                scores.append(score)
                episodes.append(e)

                if np.mean(scores[-10:]) > -20.0:
                    draw_result(episodes, scores, steps)
                    sys.exit(-1)

                if cnt % 100 == 0:
                    draw_result(episodes, scores, steps)

