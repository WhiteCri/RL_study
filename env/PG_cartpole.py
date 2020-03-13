import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from keras import backend as K
print(keras.__version__)

import sys
sys.path.append('../')
import numpy as np
import matplotlib.pyplot as plt
from running_variance import RunningVariance
import gym

#random seed
np.random.seed(1337)
n_episode = 500

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


class ActorCritic:
    def __init__(self, nS, nA, render=False, load = False):
        self.load_model = load
        self.render = render

        self.state_dim = nS
        self.action_count = nA
        self.update_frequency = 5
        self.discount_factor = 0.99
        self.running_variance = RunningVariance()

        print('state_dim: {}, action_count: {}, update_frequency: {} '.format(
        self.state_dim, self.action_count, self.update_frequency))

        #actor network
        actor = Sequential()
        actor.add(Dense(24, input_shape=(self.state_dim,), activation='relu', kernel_initializer='he_uniform'))
        actor.add(Dense(self.action_count, activation='softmax', kernel_initializer='he_uniform'))
        actor.summary()
        self.actor = actor

        #actor_optimizer
        action = K.placeholder(shape=[None, self.action_count])
        advantage = K.placeholder(shape=[None, ])

        action_prob = K.sum(action * self.actor.output, axis=1)
        cross_entropy = K.log(action_prob) * advantage
        loss = -K.sum(cross_entropy)

        optimizer = Adam(lr=0.001)
        updates = optimizer.get_updates(self.actor.trainable_weights, [], loss)
        train = K.function([self.actor.input, action, advantage], [], updates=updates)
        self.actor_optimizer = train

        #critic network
        critic = Sequential()
        critic.add(Dense(24, input_shape=(self.state_dim,), activation='relu', kernel_initializer='he_uniform'))
        critic.add(Dense(24, activation='relu', kernel_initializer='he_uniform'))
        critic.add(Dense(1, activation='linear', kernel_initializer='he_uniform'))
        critic.summary()
        self.critic = critic

        #critic optimizer
        target = K.placeholder(shape=[None, ])
        loss = K.mean(K.square(target - self.critic.output))

        optimizer = Adam(lr=0.005)
        updates = optimizer.get_updates(self.critic.trainable_weights, [], loss)
        train = K.function([self.critic.input, target], [], updates=updates)
        self.critic_optimizer = train

        if self.load_model:
            self.actor.load_weights("./cartpole_actor_185.h5")
            self.critic.load_weights("./cartpole_critic_185.h5")

    def get_action(self, state):
        policy = self.actor.predict(state, steps=1).flatten() #2d -> 1d
        return np.random.choice(self.action_count, 1, p=policy)[0]

    def train_model_1step(self, state, action, reward, next_state, done):
        value = self.critic.predict(state)[0]
        next_value = self.critic.predict(next_state)[0]

        act = np.zeros([1, self.action_count])
        act[0][action] = 1

        # 벨만 기대 방정식를 이용한 어드벤티지와 업데이트 타깃
        if done:
            advantage = reward - value
            target = [reward]
        else:
            advantage = (reward + self.discount_factor * next_value) - value
            target = reward + self.discount_factor * next_value
        self.running_variance.add(advantage)

        # debug
        r = np.square(reward + self.discount_factor * next_value - value)
        r = np.mean(r)

        self.actor_optimizer([state, act, advantage])
        self.critic_optimizer([state, target])


    def train_model(self, states, actions, rewards, dones):
        states = np.array(states)
        states = np.squeeze(states, axis=1)

        #calculate discounted reward
        discounted_rewards = self.get_discount_reward(rewards, self.discount_factor)

        #train critic network

        #train actor network using baseline
        act = np.zeros((len(actions), self.action_count), dtype=np.float32)
        act[np.arange(len(actions)), actions] = 1

        baseline = self.critic.predict(states).flatten()
        print(discounted_rewards)
        print(baseline)
        advantage = discounted_rewards - baseline
        advantage = advantage

        for r in advantage:
            self.running_variance.add(r)

        self.actor_optimizer([states, act, advantage])
        self.critic_optimizer([states, discounted_rewards])

        out = self.actor.predict(states)

        #debug
        action_prob = K.sum(act * out, axis=1)
        #print('action, out, ...', actions, out, act, K.eval(action_prob))
        cross_entropy = K.log(action_prob) * advantage
        loss = -K.sum(cross_entropy)
        #print('loss : ', K.eval(loss), K.eval(action_prob), K.eval(K.log(action_prob) * advantage))

    def get_discount_reward(self, reward, gamma=0.9):
        discounted_prediction = np.zeros_like(reward, dtype=np.float32)
        running_add = 0
        for i in reversed(range(0, len(reward))):
            running_add = running_add * gamma + reward[i]
            discounted_prediction[i] = running_add
        return discounted_prediction

def plot_results(episodes, var, scores):
    plt.subplot(121)
    plt.plot(episodes, var, 'b')
    plt.xlabel("episodes")
    plt.ylabel("variances")
    plt.title("variances")

    plt.subplot(122)
    plt.plot(episodes, scores, 'b')
    plt.xlabel("episodes")
    plt.ylabel("scores")
    plt.title("scores")
    plt.show()

if __name__=='__main__':
    #TRAIN, RENDER, LOAD = True, False, False #train
    TRAIN, RENDER, LOAD = False, True, True #evaluation

    env = gym.make('CartPole-v1') #500 max step
    nA = env.action_space.n
    nS = env.observation_space.shape[0]
    agent = ActorCritic(nS, nA, render=RENDER, load=LOAD)
    observations = Observations()
    scores, episodes, vars = [], [], []

    for e in range(n_episode):
        done = False
        score = 0
        state = env.reset()
        state = np.reshape(state, [1, nS])
        observations.reset()

        if agent.render:
            env.render()
            input()

        while not done:
            if agent.render:
                env.render()

            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)

            score += reward

            #중간에 끝나면 -100
            reward = reward if not done or score == 499 else -100

            next_state = np.reshape(next_state, [1, nS])
            observations.add(state, action, reward, done)
            if TRAIN:
                agent.train_model_1step(state, action, reward, next_state, done)

            state = next_state

            if done:
                if not TRAIN:
                    sys.exit()
                states, actions, rewards, dones = observations.get_observations()

                scores.append(score)
                episodes.append(e)
                vars.append(agent.running_variance.get_variance())

                print('episode : {}, score : {}, variacance : {}'.format(e, scores[-1], agent.running_variance.get_variance()))
                if e % 50 == 0:
                    agent.actor.save_weights("./cartpole_actor_{}.h5".format(e))
                    agent.critic.save_weights("./cartpole_critic_{}.h5".format(e))

                if np.mean(scores[-min(15, len(scores)):]) > 490:
                    agent.actor.save_weights("./cartpole_actor_{}.h5".format(e))
                    agent.critic.save_weights("./cartpole_critic_{}.h5".format(e))
                    plot_results(episodes, vars, scores)
                    sys.exit()

    plot_results(episodes, vars, scores)
