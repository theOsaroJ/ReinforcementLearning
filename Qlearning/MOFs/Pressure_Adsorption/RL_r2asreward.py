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
    prior_data['X1'] = (prior_data['X1'] - X1_mean)/X1_std

    # normalize the y
    prior_data['y'] = (prior_data['y'] - y_mean)/y_std

    # take the log scale of the X1 in test data
    test_data['X1'] = np.log10(test_data['X1'])
    test_data['y'] = np.log10(test_data['y'])
    
    # normalize the X1 and y in test data
    test_data['X1'] = (test_data['X1'] - X1_mean)/X1_std
    test_data['y'] = (test_data['y'] - y_mean)/y_std

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
        self.queried_indices = []  
        self.prior_X1 = prior_X1  
        self.prior_y = prior_y  
        self.X_test_X1 = X_test_X1
        self.y_actual_ori = y_actual_ori

        # Train the GP model with prior data if it exists
        if self.prior_X1.size > 0:
            self.gp_model.fit(self.prior_X1, self.prior_y)
            initial_predictions = self.gp_model.predict(self.prior_X1)

    def reset(self):
        self.state = 0
        self.queried_indices = []

    def step(self, action):
        next_state = action

        if next_state not in self.queried_indices:
            self.queried_indices.append(next_state)

            # Predict before adding new data point
            y_pred_before = self.gp_model.predict(self.X_test_X1)
            y_pred_before = 10**(y_pred_before * y_std + y_mean)
            #print(f"y_pred_before: {y_pred_before}")
            r2_before = r2_score(self.y_actual_ori, y_pred_before)
            mre_before = np.mean(np.abs((self.y_actual_ori - y_pred_before) / self.y_actual_ori))

            # Add the new data point and refit the GP model
            X_train_X1 = np.vstack((self.prior_X1, self.X_test_X1[self.queried_indices]))
            y_train = np.append(self.prior_y, self.y_actual[self.queried_indices])
            self.gp_model.fit(X_train_X1, y_train)

            # Predict after adding new data point
            y_pred_after = self.gp_model.predict(self.X_test_X1)
            y_pred_after = 10**(y_pred_after * y_std + y_mean)
            #print(f"y_pred_after: {y_pred_after}")
            r2_after = r2_score(self.y_actual_ori, y_pred_after)
            mre_after = np.mean(np.abs((self.y_actual_ori - y_pred_after) / self.y_actual_ori))
            
            r2_increase = r2_after - r2_before
            mre_increase = mre_after - mre_before
        else:
            r2_increase = 0.0
            mre_increase = 0.0

        done = (len(self.queried_indices) == self.num_states)

        return next_state, r2_increase, mre_increase, done

def train_q_learning(env, alpha, gamma, epsilon, max_episodes):
    Q = np.zeros((len(env.X_test_X1), len(env.X_test_X1)))

    r2_values = []
    mre_values = []

    for episode in range(max_episodes):
        env.reset()  # Reset the environment for a new episode
        state = 0
        done = False
        #print(f'Starting episode {episode}')

        while not done:
            max_q_value = np.max(Q[state, :])
            #print(f"Max Q-value at state {state}: {max_q_value}")

            rn= np.random.rand() 
            if rn < epsilon or max_q_value == 0:
                action = np.random.randint(len(env.X_test_X1))
                #print(f"Exploring @ {rn}")
            else:
                # Introduce randomness during exploitation
                action = np.random.choice(np.flatnonzero(Q[state, :] == max_q_value))
                #print(f"Exploiting @ {rn} with {max_q_value}")
            next_state, r2_increase, mre_increase, done = env.step(action)
            if action < len(env.X_test_X1):
                Q[state, action] = (1 - alpha) * Q[state, action] + alpha * (r2_increase + gamma * np.max(Q[next_state, :]))
            state = next_state

        X1_test_episode = env.X_test_X1[action]
        y_actual_episode = env.y_actual_ori[action]
        y_actual_episode = (np.log10(y_actual_episode) - y_mean) / y_std
        
        # Update prior data for the next episode
        env.prior_X1 = np.append(env.prior_X1, env.X_test_X1[action]).reshape(-1, 1)
        env.prior_y = np.append(env.prior_y, y_actual_episode)
        env.gp_model.fit(np.array(env.prior_X1).reshape(-1, 1), env.prior_y)

        predicted_values = env.gp_model.predict(env.X_test_X1)
        predicted_values = 10**(predicted_values * y_std + y_mean)
        r2_final = r2_score(env.y_actual_ori, predicted_values)
        r2_values.append(r2_final)

        mre_final = np.mean(np.abs((env.y_actual_ori - predicted_values) / env.y_actual_ori))
        mre_values.append(mre_final)

        print(f'Episode {episode} - R2 Score: {r2_final:.4f} - MRE: {mre_final:.4f}')

    # return the r2 values and mre values but concatenate them into a 2D array
    values = np.concatenate((np.array(r2_values).reshape(-1, 1), np.array(mre_values).reshape(-1, 1)), axis=1)

    return values
   
param_grid = {
    'alpha': [0.1],
    'gamma': [0.9],
    'epsilon': [0.1],
    'max_episodes': [10],
    'kernel': [RationalQuadratic(length_scale=50, alpha=0.5, length_scale_bounds=(1e-13, 1e13), alpha_bounds=(1e-13, 1e13))]
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

best_mre = float('inf')
best_mre_params = None
best_mre_values = None

for params in ParameterGrid(param_grid):
    print(params)
    gp = GaussianProcessRegressor(kernel=params['kernel'], n_restarts_optimizer=50, normalize_y=True)
    
    # Create the environment with the trained GP model
    env = Environment(X_test_X1_ori, gp, y_actual, prior_X1, prior_y, X_test_X1)

    values = train_q_learning(env, params['alpha'], params['gamma'], params['epsilon'], params['max_episodes'])

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
    np.savetxt(mre_values_filename, values[:,1], delimiter=',')

    if values[-1][0] > best_r2:
        best_r2 = values[-1][0]
        best_r2_params = params
        best_values = values

    if values[-1][1] < best_mre:
        best_mre = values[-1][1]
        best_mre_params = params
        best_mre_values = values

print("Best R2 Hyperparameters:")
print(best_r2_params)
print("Best R2 Score:", best_r2)
print("Best MRE Score:", best_mre)
