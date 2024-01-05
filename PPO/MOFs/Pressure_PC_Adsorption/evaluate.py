from gpr_env import GPR_Env
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
import csv

def mask_fn(env):
    return env.valid_action_mask()

batch_size = 10

# dataset path
prior_path = '/scratch365/eosaro/Research/DeepRL/Bagging/bag_1/dataset/Prior.csv'
test_path = '/scratch365/eosaro/Research/DeepRL/Bagging/bag_1/dataset/Test.csv'

# path to the .zip model file (GPR_PPO.zip)
# REMEMBER: don't add .zip at the end of the file    
model_path = './model/PPO_GPR'  

# path to save the r2 and prior of each episode
record_file = '/scratch365/eosaro/Research/DeepRL/Bagging/bag_1/results/action_list.csv'     

# best prior path
best_eval_prior_path = '/scratch365/eosaro/Research/DeepRL/Bagging/bag_1/results/best_prior_eval.csv'     

env = GPR_Env(prior_path, test_path, best_eval_prior_path, batch_size, train=False)
env = ActionMasker(env, mask_fn)

model = MaskablePPO.load(model_path, env=env)
num_episodes = 20

with open(record_file, 'a') as f:
    header = ['eps', 'length', 'r2_list', 'action_list']
    writer_obj = csv.writer(f)
    writer_obj.writerow(header)

for eps in range(num_episodes):
    done = False
    obs, info = env.reset()
    r2_list = [info['r2']]
    action_list = []
    while not done:
        action, _ = model.predict(obs, action_masks=obs)
        action = action.item(0)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        r2_list.append(info['r2'])
        action_list.append(action)
    with open(record_file, 'a') as f:
        writer_obj = csv.writer(f)
        writer_obj.writerow([eps, len(action_list), r2_list, action_list])
    print(f'Episode {eps} - Length action = {len(action_list)}')

env.close()
