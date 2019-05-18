import numpy as np
import gym
import collections
import logging
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


def export_plot(ys, ylabel, filename):
    plt.figure()
    plt.plot(range(len(ys)), ys, color='crimson')
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.savefig(filename)
    plt.close()

def get_logger(filename):
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format='%(message)s', level=logging.DEBUG)
    handler = logging.FileHandler(filename)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logging.getLogger().addHandler(handler)
    return logger

class LinearSchedule(object):
    def __init__(self, eps_begin, eps_end, nsteps):
        self.epsilon        = eps_begin
        self.eps_begin      = eps_begin
        self.eps_end        = eps_end
        self.nsteps         = nsteps

    def update(self, t):
        decay = np.linspace(self.eps_begin, self.eps_end, self.nsteps+1)
        if t > self.nsteps:
            self.epsilon = self.eps_end
        else:
            self.epsilon = decay[t]
            
class LinearExploration(LinearSchedule):
    def __init__(self, env, eps_begin, eps_end, nsteps):
        self.env = env
        super(LinearExploration, self).__init__(eps_begin, eps_end, nsteps)


    def get_action(self, best_action):
        """
        Returns a random action with prob epsilon, otherwise returns the best_action
        """
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return best_action
        
def greyscale(state):
    """
    Preprocess (210, 160, 3) image into (80, 80, 1) greyscale
    """
    state = np.reshape(state, [210, 160, 3]).astype(np.float32)
    state = state[:, :, 0] * 0.299 + state[:, :, 1] * 0.587 + state[:, :, 2] * 0.114
    state = state[35:195]  # crop to (160, 160, 1)
    state = state[::2,::2] # downsample by factor of 2 to (80, 80, 1)
    state = state[:, :, np.newaxis]
    return state.astype(np.uint8)

class MaxAndSkipWrapper(gym.Wrapper):
    '''
    modified from https://github.com/openai/atari-reset/blob/master/atari_reset/wrappers.py
    '''
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        super(MaxAndSkipWrapper, self).__init__(env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = collections.deque(maxlen=2)
        self._skip       = skip

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        combined_info = {}
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            combined_info.update(info)
            if done:
                break
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)

        return max_frame, total_reward, done, combined_info

    def reset(self):
        """Clear past frame buffer and init. to first obs. from inner env."""
        self._obs_buffer.clear()
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs

class ResizeWrapper(gym.Wrapper):
    def __init__(self, env, preprocess, shape):
        super(ResizeWrapper, self).__init__(env)
        # most recent raw observations (for max pooling across time steps)
        self.prep = preprocess
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=shape, dtype = np.uint8)

    def reset(self):
        return self.prep(self.env.reset())

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self.prep(obs), reward, done, info