import gymnasium as gym
from gymnasium import spaces
import numpy as np
from gpmodel_class import GPModel
from csv import writer

class GPR_Env(gym.Env):
    def __init__(self, prior_file='Prior.csv', test_file='Test.csv', best_dataset_path='best_prior_train.csv', batch_size=10, train=True):
        self.gpmodel = GPModel(prior_file, test_file, batch_size)
        self.MAX_DATA_POINTS = self.gpmodel.data_length
        self.MAX_STEP = self.MAX_DATA_POINTS - 1
        self.action_space = spaces.Discrete(self.MAX_DATA_POINTS, seed=40)
        self.observation_space = spaces.MultiBinary(self.MAX_DATA_POINTS, seed=40)
        self.best_dataset_path = best_dataset_path
        self.train = train
    
    def step(self, action):
        truncated = False
        terminated = False
        self.remained_datapoints.append(action)
        if self._step == self.MAX_STEP:
            truncated = True
            reward = -10
        # print(f'Current state at action {action} = {self.current_state[action]}')
        self.current_state[action] = False
        # print(f'Current state at action {action} = {self.current_state[action]}')
        self.gpmodel.update_model(action)
        
        r2 = self.gpmodel.calculate_r2()
        info = {'r2': r2}
        if r2 >= 0.985:
            terminated = True
            reward = 20
            if self.train:
                if self._step < self.MAX_STEP:
                    self.MAX_STEP = self._step
            with open(self.best_dataset_path, 'a') as best_ds:
                row = [r2, len(self.remained_datapoints), self.remained_datapoints]
                write_obj = writer(best_ds)
                write_obj.writerow(row)
        elif r2 < -10.0:
            reward = -10
            self._prev_r2 = -10.0
        else:
            reward = 0.01 * (r2 - self._prev_r2)
            self._prev_r2 = r2

        self._step += 1
        return self.current_state, reward, terminated, truncated, info
    
    def reset(self, seed=None, options=None):
        self.remained_datapoints = []
        
        self.current_state = np.ones(self.MAX_DATA_POINTS, dtype=bool)
        self._step = 0
        
        self.gpmodel.reset_dataset()
        self.gpmodel.update_model(None)
        self._prev_r2 = self.gpmodel.calculate_r2()
        info = {'r2': self._prev_r2}
        
        return self.current_state, info
    
    @property
    def r2(self):
        return self._prev_r2
    
    @property
    def max_data_points(self):
        return self.MAX_STEP
        
    def render(self):
        pass
    
    def close(self):
        return super().close()
    
    def valid_action_mask(self):
        return self.current_state
