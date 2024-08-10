import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RationalQuadratic, Matern
from sklearn.metrics import r2_score
from sklearn.model_selection import ParameterGrid
import os
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
import warnings
warnings.filterwarnings("ignore")
import time
import signal

np.random.seed(42)

# Load X_test data from the CSV file
test_data = pd.read_csv('Test.csv', delimiter=',')

X_test_X1_ori = test_data['X1'].values.reshape(-1, 1)
y_actual_ori = test_data['y'].values

# Load the training data from 'Prior.csv'
prior_data = pd.read_csv('Prior.csv', delimiter=',') if 'Prior.csv' in os.listdir() else None

# Initialize prior data or use an empty DataFrame if 'Prior.csv' doesn't exist
if prior_data is not None:
    # take the log scale of the X1 and y
    prior_data['X1'] = np.log10(prior_data['X1'])
    prior_data['y'] = np.log10(prior_data['y'])

    # calculate the mean and std of the X1 and y
    X1_mean = prior_data['X1'].mean()
    X1_std = prior_data['X1'].std()

    y_mean = prior_data['y'].mean()
    y_std = prior_data['y'].std()

    # normalize the X1
    prior_data['X1'] = (prior_data['X1'] - X1_mean) / X1_std

    # normalize the y
    prior_data['y'] = (prior_data['y'] - y_mean) / y_std

    # take the log scale of the X1 in test data
    test_data['X1'] = np.log10(test_data['X1'])
    test_data['y'] = np.log10(test_data['y'])
    
    # normalize the X1 and y in test data
    test_data['X1'] = (test_data['X1'] - X1_mean) / X1_std
    test_data['y'] = (test_data['y'] - y_mean) / y_std

    prior_X1 = prior_data['X1'].values.reshape(-1, 1)
    prior_y = prior_data['y'].values
    
else:
    prior_X1 = np.array([]).reshape(-1, 1)
    prior_y = np.array([])

X_test_X1 = test_data['X1'].values.reshape(-1, 1)
y_actual = test_data['y'].values

class Environment:
    def __init__(self, X_test_X1_ori, gp_model, y_actual, prior_X1, prior_y, X_test_X1):
        self.X_test_X1_ori = X_test_X1_ori
        self.gp_model = gp_model
        self.y_actual = y_actual
        self.num_states = len(X_test_X1_ori) 
        self.num_actions = len(X_test_X1_ori)  
        self.state = 0  
        self.queried_indices = set()  # Use a set for faster lookup
        self.prior_X1 = prior_X1  
        self.prior_y = prior_y  
        self.X_test_X1 = X_test_X1
        self.y_actual_ori = y_actual_ori

        # Train the GP model with prior data if it exists
        if self.prior_X1.size > 0:
            self.gp_model.fit(self.prior_X1, self.prior_y)
            print("Initial model fit with prior data")
            initial_predictions = self.gp_model.predict(self.prior_X1)

    def reset(self):
        self.state = 0
        self.queried_indices.clear()

    def additional_check(self, X1_new, y_new):
        if X1_new not in self.prior_X1:
            self.prior_X1 = np.append(self.prior_X1, X1_new).reshape(-1, 1)
            self.prior_y = np.append(self.prior_y, y_new)
        else:
            for i in range(len(self.X_test_X1)):
                if self.X_test_X1[i] not in self.prior_X1:
                    self.prior_X1 = np.append(self.prior_X1, self.X_test_X1[i]).reshape(-1, 1)
                    self.prior_y = np.append(self.prior_y, self.y_actual[i])
                    self.X_test_X1 = np.delete(self.X_test_X1, i, axis=0)
                    self.y_actual = np.delete(self.y_actual, i)
                    self.y_actual_ori = np.delete(self.y_actual_ori, i)
                    break

    def step(self, action):
        next_state = action

        if next_state not in self.queried_indices:
            self.queried_indices.add(next_state)

            try:
                # Predict before adding new data point
                y_pred_before = self.gp_model.predict(self.X_test_X1)
                y_pred_before = 10**(y_pred_before * y_std + y_mean)
                r2_before = r2_score(self.y_actual_ori, y_pred_before)
                mre_before = np.mean(np.abs((self.y_actual_ori - y_pred_before) / self.y_actual_ori))

                # Add the new data point and refit the GP model
                valid_indices = list(self.queried_indices)
                X_train_X1 = np.vstack((self.prior_X1, self.X_test_X1[valid_indices]))
                y_train = np.append(self.prior_y, self.y_actual[valid_indices])
                self.gp_model.fit(X_train_X1, y_train)

                # Predict after adding new data point
                y_pred_after = self.gp_model.predict(self.X_test_X1)
                y_pred_after = 10**(y_pred_after * y_std + y_mean)
                r2_after = r2_score(self.y_actual_ori, y_pred_after)
                mre_after = np.mean(np.abs((self.y_actual_ori - y_pred_after) / self.y_actual_ori))
                
                r2_increase = r2_after - r2_before
                mre_increase = mre_after - mre_before
            except IndexError:
                raise 
        else:
            r2_increase = 0.0
            mre_increase = 0.0

        done = (len(self.queried_indices) == self.num_states)

        return next_state, r2_increase, mre_increase, done

def train_q_learning(env, alpha, gamma, epsilon, max_episodes):
    Q = np.zeros((len(env.X_test_X1), len(env.X_test_X1)))

    r2_values = []
    mre_values = []

    try:
        for episode in range(max_episodes):
            env.reset()  # Reset the environment for a new episode
            state = 0
            done = False
            while not done:
                max_q_value = np.max(Q[state, :])
                if np.random.rand() < epsilon or max_q_value == 0:
                    action = np.random.randint(len(env.X_test_X1))
                else:
                    # Introduce randomness during exploitation
                    action = np.random.choice(np.flatnonzero(Q[state, :] == max_q_value))
                
                # Ensure action is not repeated
                while action in env.queried_indices:
                    action = np.random.randint(len(env.X_test_X1))
                    
                next_state, r2_increase, mre_increase, done = env.step(action)

                if action < len(env.X_test_X1):
                    Q[state, action] = (1 - alpha) * Q[state, action] + alpha * (mre_increase + gamma * np.max(Q[next_state, :]))

                state = next_state

            X1_test_episode = env.X_test_X1[action]
            y_actual_episode = env.y_actual_ori[action]
            y_actual_episode = (np.log10(y_actual_episode) - y_mean) / y_std
            
            # Additional check before updating prior data
            env.additional_check(X1_test_episode, y_actual_episode)

            # Refit the GP model with the updated prior data
            env.gp_model.fit(np.array(env.prior_X1).reshape(-1, 1), env.prior_y)

            predicted_values = env.gp_model.predict(env.X_test_X1)
            predicted_values = 10**(predicted_values * y_std + y_mean)
            r2_final = r2_score(env.y_actual_ori, predicted_values)
            r2_values.append(r2_final)

            mre_final = np.mean(np.abs((env.y_actual_ori - predicted_values) / env.y_actual_ori))
            mre_values.append(mre_final)

            print(f'Episode {episode} - R2 Score: {r2_final:.4f} - MRE: {mre_final:.4f}')
    except IndexError:
        print("IndexError encountered, moving to the next parameter grid.")

    # return the r2 values and mre values but concatenate them into a 2D array
    values = np.concatenate((np.array(r2_values).reshape(-1, 1), np.array(mre_values).reshape(-1, 1)), axis=1)

    return values
param_grid = {
    'alpha': [0.001, 0.01, 0.1],
    'gamma': [0.1, 0.5, 0.9],
    'epsilon': [0.1, 0.5, 0.9],
    'max_episodes': [10],
    'kernel': [RationalQuadratic(length_scale=1, alpha=0.5, length_scale_bounds=(1e-13, 1e13), alpha_bounds=(1e-13, 1e13))]
}

def get_kernel_abbreviation(kernel):
    if isinstance(kernel, RationalQuadratic):
        return 'RQ'
    else:
        return 'Other'

def save_data(prior_X1, prior_y, predicted_values, alpha, gamma, epsilon, max_episodes, kernel):
    kernel_abbr = get_kernel_abbreviation(kernel)

    combined_data = np.hstack((prior_X1.reshape(-1, 1), prior_y.reshape(-1, 1)))
    filename = f'Prior_data_{alpha}_{gamma}_{epsilon}_{max_episodes}_{kernel_abbr}.csv'
    np.savetxt(filename, combined_data, delimiter=',')

    predicted_filename = f'Predicted_values_{alpha}_{gamma}_{epsilon}_{max_episodes}_{kernel_abbr}.csv'
    np.savetxt(predicted_filename, predicted_values, delimiter=',')

best_r2 = -float('inf')
best_r2_params = None
best_r2_values = None

# get the best MRE as the best MRE is the lowest value
best_mre = float('inf')
best_mre_params = None
best_mre_values = None

def handler(signum, frame):
    raise TimeoutError

signal.signal(signal.SIGALRM, handler)

for params in ParameterGrid(param_grid):
    print(params)
    gp = GaussianProcessRegressor(kernel=params['kernel'], n_restarts_optimizer=50, normalize_y=True)

    # Create the environment with the trained GP model
    env = Environment(X_test_X1_ori, gp, y_actual, prior_X1, prior_y, X_test_X1)

    try:
        signal.alarm(360)  # Set the timeout to 60 seconds
        values = train_q_learning(env, params['alpha'], params['gamma'], params['epsilon'], params['max_episodes'])
        signal.alarm(0)  # Disable the alarm
    except TimeoutError:
        print(f"Timeout for parameter set {params}, moving to the next set.")
        continue
    except IndexError:
        continue

    predicted_values = env.gp_model.predict(X_test_X1)
    predicted_values = 10**((predicted_values * y_std) + y_mean)
    predicted_values = predicted_values.tolist()
    for i in range(len(predicted_values)):
        if predicted_values[i] <= 0:
            predicted_values[i] = 1e-5

    # convert the data back to original scale
    env.prior_X1 = 10**((env.prior_X1 * X1_std) + X1_mean)
    env.prior_y = 10**((env.prior_y * y_std) + y_mean)

    save_data(env.prior_X1, env.prior_y, predicted_values, params['alpha'], params['gamma'], params['epsilon'], params['max_episodes'], params['kernel'])

    r2_values_filename = f'R2_values_{params["alpha"]}_{params["gamma"]}_{params["epsilon"]}_{params["max_episodes"]}_{get_kernel_abbreviation(params["kernel"])}.csv'
    np.savetxt(r2_values_filename, values[:,0], delimiter=',')

    mre_values_filename = f'MRE_values_{params["alpha"]}_{params["gamma"]}_{params["epsilon"]}_{params["max_episodes"]}_{get_kernel_abbreviation(params["kernel"])}.csv'
    np.savetxt(mre_values_filename, values[:, 1], delimiter=',')

    if values[-1][0] > best_r2:
        best_r2 = values[-1][0]
        best_r2_params = params
        best_r2_values = values
        best_r2_prior_X1 = env.prior_X1
        best_r2_prior_y = env.prior_y
        best_r2_predicted_values = predicted_values

    if values[-1][1] < best_mre:
        best_mre = values[-1][1]
        best_mre_params = params
        best_mre_values = values
        best_mre_prior_X1 = env.prior_X1
        best_mre_prior_y = env.prior_y
        best_mre_predicted_values = predicted_values

print("Best R2 Hyperparameters:")
print(best_r2_params)
print("Best R2 Score:", best_r2)
print("Best MRE Hyperparameters:")
print(best_mre_params)
print("Best MRE Score:", best_mre)

# Save the best R2 evolution
if best_r2_values is not None:
    best_r2_values_filename = f'Final_R2_values.csv'
    np.savetxt(best_r2_values_filename, best_r2_values[:, 0], delimiter=',')

    best_r2_prior_data = pd.DataFrame({'X1': best_r2_prior_X1.flatten(), 'y': best_r2_prior_y})
    best_r2_prior_data_filename = f'Final_R2_prior.csv'
    best_r2_prior_data.to_csv(best_r2_prior_data_filename, index=False)

# Save the best MRE evolution
if best_mre_values is not None:
    best_mre_values_filename = f'Final_MRE_values.csv'
    np.savetxt(best_mre_values_filename, best_mre_values[:, 1], delimiter=',')

    best_mre_prior_data = pd.DataFrame({'X1': best_mre_prior_X1.flatten(), 'y': best_mre_prior_y})
    best_mre_prior_data_filename = f'Final_MRE_prior.csv'
    best_mre_prior_data.to_csv(best_mre_prior_data_filename, index=False)
