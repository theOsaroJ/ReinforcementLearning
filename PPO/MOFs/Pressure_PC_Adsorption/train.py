from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO

from gpr_env import GPR_Env

config = {
    'GPR_batch': 10,
    'total_timesteps': 2000,
    'n_steps': 100,
    'checkpoint': 100
}

class TensorboardCallbacks(BaseCallback):
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        
    def _on_step(self):
        self.logger.record('info/r2', self.model.env.get_attr('r2')[0])
        self.logger.record('info/Max steps', self.model.env.get_attr('max_data_points')[0])
        return True 

def mask_fn(env):
    return env.valid_action_mask()

# path to dataset
def train(wandb_usage=False):
    prior_path = '/scratch365/eosaro/Research/DeepRL/Bagging/bag_1/dataset/Prior.csv'
    test_path = '/scratch365/eosaro/Research/DeepRL/Bagging/bag_1/dataset/Test.csv'
    best_train_prior_path = '/scratch365/eosaro/Research/DeepRL/Bagging/bag_1/results/best_prior_train.csv'

    env = GPR_Env(prior_path, test_path, best_dataset_path=best_train_prior_path, batch_size=config['GPR_batch'])  # Initialize env
    # env = GPR_Env()
    env = ActionMasker(env, mask_fn)  # Wrap to enable masking
    env = DummyVecEnv([lambda: Monitor(env)])

    model = MaskablePPO(MaskableActorCriticPolicy, env, verbose=1, tensorboard_log='./tb_log', n_steps=config['n_steps'])
    print('Start training')
    if wandb_usage:
        import wandb
        from wandb.integration.sb3 import WandbCallback
        
        run = wandb.init(project='GPR_model_RL', name='test-1', entity='gpr-online', sync_tensorboard=True, save_code=False, config=config)
    
        model.learn(total_timesteps=wandb.config['total_timesteps'], callback=[CheckpointCallback(wandb.config['checkpoint'], save_path='./checkpoint'), 
                                                                        TensorboardCallbacks(),
                                                                        WandbCallback(gradient_save_freq=100, model_save_path='models/run_' + str(run.id))])
    else:
        model.learn(total_timesteps=config['total_timesteps'], callback=[CheckpointCallback(config['checkpoint'], save_path='./checkpoint'), 
                                                                        TensorboardCallbacks()])

    model.save('./model/PPO_GPR')

    env.close()
    
if __name__ == '__main__':
    train()
