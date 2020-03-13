import numpy as np
import sys
from gym.envs.toy_text import discrete
from io import StringIO

# Cliffwalking environment

class CliffWalkingEnv(discrete.DiscreteEnv):
    metadata = {'render.modes' : {'human', 'ansi', 'rgb_array'}}

    '''
        state : 0~47
        
        
        coordinate : (x, y)
         0-------y---->
         l
         x
         l
        \ /
    '''

    def __init__(self):
        self.shape = (6, 14)
        self.action_to_delta = [(-1,0), (0, 1), (1, 0), (0, -1)]
        nS = np.prod(self.shape)
        nA = 4 #0, 1, 2, 3: UP, RIGHT, DOWN, LEFT

        self._cliff = np.zeros(self.shape, dtype=np.bool)
        self._cliff[0, :] = True
        self._cliff[-1, :] = True
        self._cliff[:, 0] = True
        self._cliff[:, -1] = True
        self._cliff[4, 2:-2] = True

        self.n_step = 0
        #Calculate transitions
        P = {}
        for s in range(nS):
            position = np.unravel_index(s, self.shape)
            P[s] = {a : [] for a in range(nA)}
            P[s][0] = self._calculate_transition_prob(position, 0)
            P[s][1] = self._calculate_transition_prob(position, 1)
            P[s][2] = self._calculate_transition_prob(position, 2)
            P[s][3] = self._calculate_transition_prob(position, 3)

        isd = np.zeros(nS) #initial state distribution
        isd[np.ravel_multi_index((4, 1), self.shape)] = 1.0

        super(CliffWalkingEnv, self).__init__(nS, nA, P, isd)

    def _limit_coordinates(self, coord):
        coord[0] = min(coord[0], self.shape[0] - 1)
        coord[0] = max(coord[0], 0)
        coord[1] = min(coord[1], self.shape[1] - 1)
        coord[1] = max(coord[1], 0)
        return coord

    def _calculate_transition_prob(self, current_coord, action):
        delta = self.action_to_delta[action]
        new_position = np.array(current_coord) + np.array(delta)
        new_position = self._limit_coordinates(new_position)
        new_state = np.ravel_multi_index(tuple(new_position), self.shape)
        reward = -100.0 if self._cliff[tuple(new_position)] else 0
        if tuple(new_position) == (4, 12):
            reward = 100
        is_done = self._cliff[tuple(new_position)] or (tuple(new_position) == (4, 12))
        reward = -100.0 if self._cliff[tuple(new_position)] else 0

        return [(1.0, new_state, reward, is_done)]

    def _convert_state(self, state): # () -> np ary coordinate
        converted = np.unravel_index(state, self.shape)
        return np.asarray(list(converted), dtype=np.int)

    def reset(self):
        self.s = np.argmax(self.isd)
        self.n_step = 0
        return self._convert_state(self.s)

    def step(self, action):
        self.n_step += 1
        done = self.P[self.s][action][0][3] or (self.n_step == 100)
        reward = -100.0 if self.n_step == 100 else self.P[self.s][action][0][2]
        info = {'prob': self.P[self.s][action][0][1]}
        self.s = self.P[self.s][action][0][1]
        return (self._convert_state(self.s), reward, done, info)

    def render(self, mode='rgb_array', close=False):
        if close:
            return
        if mode == 'rgb_array':
            maze = np.zeros(self.shape)
            maze[self._cliff] = -1
            maze[np.unravel_index(self.s, self.shape)] = 2.0
            maze[(3,11)] = 0.5
            img = np.array(maze, copy=True)
            return img
        else:
            outfile = StringIO() if mode == 'ansi' else sys.stdout

            for s in range(self.nS):
                position = np.unravel_index(s, self.shape)

                if self.s == s:
                    output = " x "
                elif position == (4, 12):
                    output = " T "
                elif self._cliff[position]:
                    output = " C "
                else:
                    output = " o "

                if position[1] == 0:
                    output = output.lstrip()
                if position[1] == self.shape[1] - 1:
                    output = output.rstrip()
                    output += '\n'

                outfile.write(output)

            outfile.write('\n')


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import time
    env = CliffWalkingEnv()
    action_ls = [0,1,1,1,1,1,1,1,1,1,1,1,2]
    for a in action_ls:
        img = env.render()

        plt.imshow(img)
        plt.draw()
        plt.pause(0.5)
        env.step(a)

    img = env.render()
    plt.imshow(img)
    plt.show()

