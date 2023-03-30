import gym
import numpy as np
import pandas as pd
from gym import spaces

# Class from the Introduction chapter
class Segment(gym.Env):
    """
    Generate a simple segment of grids to navigate

    Attributes:
        num_actions (int): the number of possible actions
        observation (int): the observed state
        p (float): Probability of moving left
        terminal (int): a terminal state
        action_space (Dict): a mapping of actions
        done (Bool): Whether and episode has finished playing out

    Methods:
        step(action)
        rewards(observation, observation_next, action)
        reset()
    """
    def __init__(self, num_actions, start_observation, p, terminal, done):
        self.num_actions = num_actions
        self.observation = start_observation
        self.p = p
        self.terminal = terminal

        # {0:left, 1:right}
        self.action_space = spaces.Discrete(self.num_actions) 
        self.observation_space = spaces.Discrete(2 * self.terminal + 1)
        self.done = done

    def step(self, action):
        """
        Update the obseration and the next observation based on an action

        Parameters:
            action (int): an integer representing the action taken

        Returns:
            tuple (int, int, Bool): observed state, reward, whether finished
        """
        assert self.action_space.contains(action)
        assert self.observation_space.contains(self.observation)
        assert self.action_space.n == 2
        assert self.observation != 0
        assert self.observation != (2 * self.terminal)
        observation = self.observation
        done = self.done

        if action == 0:
            observation_next = observation + np.random.choice([-1, 1], 
                                                              p=[self.p, 
                                                                 1 - self.p])
        elif action == 1:
            observation_next = observation + np.random.choice([-1, 1], 
                                                              p=[1 - self.p, 
                                                                 self.p])
        if observation_next == (2 * self.terminal):
            done = True
        elif observation_next == 0:
            done = True
        self.done = done
        self.observation = observation_next
        reward = self.rewards(observation, observation_next, action)

        return self.observation, reward, done

    def rewards(self, observation, observation_next, action):
        """
        Generate a reward for a state, the next state, and an action

        Parameters:
            observation (int): the observed state
            observation_next (int): the next state
            action (int): the action taken

        Returns:
            float: the reqard given the state, next state, and action
        """
        if observation_next==(2 * self.terminal):
            reward = 1.0
        elif observation_next == 0:
            reward = -1.0
        else:
            reward = -0.05
        return reward

    def reset(self):
        """
        Reset the environment
        """
        observation = self.startObservation
        self.observation = observation
        self.done = False

        return observation
