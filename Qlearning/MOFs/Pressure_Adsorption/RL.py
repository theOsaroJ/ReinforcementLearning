'''
Authors: Etinosa Osaro, Yamil J Colon
'''
#!/usr/bin/env python3
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

# Load X_test data from the CSV file
test_data = pd.read_csv('Test.csv', delimiter=',')

# Load the training data from 'Prior.csv'
prior_data = pd.read_csv('Prior.csv', delimiter=',') if 'Prior.csv' in os.listdir() else None

# Initialize prior data or use an empty DataFrame if 'Prior.csv' doesn't exist
if prior_data is not None:
    prior_X = prior_data['X'].values.reshape(-1, 1)
    prior_y = prior_data['y'].values
else:
    prior_X = np.array([])
    prior_y = np.array([])

X_test = test_data['X'].values.reshape(-1,1)
y_actual = test_data['y'].values

# Define the RL environment
class Environment:
    def __init__(self, X_test, gp_model, y_actual, prior_X, prior_y):
        self.X_test = X_test
        self.gp_model = gp_model
        self.y_actual = y_actual
        self.num_states = len(X_test)  # Number of states (data points in X_test)
        self.num_actions = len(X_test)  # Number of actions (select any data point)
        self.state = 0  # Initial state (start from the first data point)
        self.queried_indices = []  # To keep track of queried data points
        self.prior_X = prior_X  # Initialize prior data
        self.prior_y = prior_y  # Initialize prior target values

    def step(self, action):
        # Simulate acquiring data (labeling) for the selected data point
        next_state = action

        if next_state not in self.queried_indices:
            self.queried_indices.append(next_state)
            # Fit the GP model
            X_train = self.X_test[self.queried_indices]
            y_train = self.y_actual[self.queried_indices]
            self.gp_model.fit(X_train, y_train)
            r2_before = r2_score(self.y_actual, self.gp_model.predict(self.X_test))
            
            if len(self.queried_indices) > 1 and len(self.queried_indices) <= len(X_train):
                r2_after = r2_score(self.y_actual, self.gp_model.predict(self.X_test))
                r2_increase = r2_after - r2_before  # Reward is the increase in R2 score
            else:
                r2_increase = -1.0  # A penalty for querying the same data point again
        else:
            r2_increase = -1.0  # A penalty for querying the same data point again

        done = (len(self.queried_indices) == self.num_states)  # Check if all data points have been queried

        return next_state, r2_increase, done

# Function to train the Q-learning agent with given hyperparameters
def train_q_learning(env, gp, alpha, gamma, epsilon, max_episodes):
    Q = np.zeros((len(env.X_test), len(env.X_test)))
    r2_values = []

    for episode in range(max_episodes):
        state = 0  # Set the initial state as 0 (or any other appropriate initial state)
        done = False

        while not done:
            if np.random.rand() < epsilon:
                action = np.random.randint(len(env.X_test))  # Exploration: choose a random action
            else:
                action = np.argmax(Q[state, :])  # Exploitation: choose the action with the highest Q-value

            next_state, r2_increase, done = env.step(action)

            if action < len(env.X_test):  # Check for valid indices
                # Q-learning update rule
                Q[state, action] = (1 - alpha) * Q[state, action] + alpha * (r2_increase + gamma * np.max(Q[next_state, :]))
            else:
                r2_increase = -1.0  # A penalty for invalid actions

            state = next_state

        X_test_episode = env.X_test[action]
        y_actual_episode = env.y_actual[action]
        print(f'Added Data - X_test episode: {X_test_episode} - y_actual episode: {y_actual_episode}')
        
        # Update prior data for the next episode
        env.prior_X = np.append(env.prior_X, env.X_test[action])
        env.prior_y = np.append(env.prior_y, env.y_actual[action])
        print('Data')
        print(env.prior_X)
        print(env.prior_y)

        # Refit the GP model with the updated prior data
        env.gp_model.fit(np.array(env.prior_X).reshape(-1, 1), env.prior_y)

        # compute the predicted values

        print(f"Predicted value: {env.gp_model.predict(env.X_test)}")
        # Calculate the final R2 after each episode and store it
        r2_final = r2_score(env.y_actual, env.gp_model.predict(env.X_test))
        r2_values.append(r2_final)

        # Print the R2 score after each episode
        print(f'Episode {episode} - R2 Score: {r2_final:.4f}')


        # Stop episodes if R2 score >= 0.99
        if r2_final > 0.985:
            print('A break happened')
            break

    return r2_values

param_grid = {
    'alpha': [0.001, 0.01, 0.1],
    'gamma': [0.1, 0.3, 0.5, 0.7, 0.9],
    'epsilon': [0.1, 0.3, 0.5, 0.7, 0.9],
    'max_episodes': [5, 10, 15],
    'kernel':[RationalQuadratic(length_scale=50, alpha=0.5,length_scale_bounds=(1e-8,1e8),alpha_bounds=(1e-8,1e8)), Matern(length_scale=50, nu=1.5)]
}


# Function to get kernel abbreviation based on type
def get_kernel_abbreviation(kernel):
    if isinstance(kernel, RationalQuadratic):
        return 'RQ'
    elif isinstance(kernel, Matern):
        return 'MA'
    else:
        return 'Other'

# Function to save data to a file with specific parameters
def save_data(prior_X, prior_y, predicted_values, alpha, gamma, epsilon, max_episodes, kernel):
    # Abbreviate kernel type
    kernel_abbr = get_kernel_abbreviation(kernel)
    
    # Save prior X and prior y
    combined_data = np.hstack((prior_X.reshape(-1, 1), prior_y.reshape(-1, 1)))
    filename = f'Prior_data_{alpha}_{gamma}_{epsilon}_{max_episodes}_{kernel_abbr}.csv'
    np.savetxt(filename, combined_data, delimiter=',')

    # Save predicted values
    predicted_filename = f'Predicted_values_{alpha}_{gamma}_{epsilon}_{max_episodes}_{kernel_abbr}.csv'
    np.savetxt(predicted_filename, predicted_values, delimiter=',')

best_r2 = -float('inf')
best_params = None

for params in ParameterGrid(param_grid):
    print(params)
    # Create the Q-learning environment and GP model with the current parameters
    gp = GaussianProcessRegressor(kernel=params['kernel'], n_restarts_optimizer=50, normalize_y=True)
    env = Environment(X_test, gp, y_actual, prior_X, prior_y)

    # Train the Q-learning agent
    r2_values = train_q_learning(env, gp, params['alpha'], params['gamma'], params['epsilon'], params['max_episodes'])

    # Get predicted values after training
    predicted_values = env.gp_model.predict(env.X_test)
    predicted_values = predicted_values.tolist()
    for i in range(len(predicted_values)):
        if (predicted_values[i] <= 0):
                predicted_values[i] = 1e-5

    # Save data to files with specific parameters
    save_data(env.prior_X, env.prior_y, predicted_values, params['alpha'], params['gamma'], params['epsilon'], params['max_episodes'], params['kernel'])

    # Check if the current parameters result in a better R2 score
    if r2_values[-1] > best_r2:
        best_r2 = r2_values[-1]
        best_params = params

print("Best Hyperparameters:")
print(best_params)
print("Best R2 Score:", best_r2)
