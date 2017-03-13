# -----------------------------------------------------------------------------
#   @brief:
#       In this file, all the parameters of the network is configurated
#       For different network, we use separate parameters
#       Long life easydict
# -----------------------------------------------------------------------------


from easydict import EasyDict as edict
from network_config import network_config

__C = edict()
base_config = __C

__C.NETWORK = network_config
__C.ATARI = edict()
__C.EXPERIENCE = edict()
__C.TRAIN = edict()
__C.GAME = edict()

# basic training parameters
__C.TRAIN.batch_size = 32
__C.TRAIN.gradient_clip = 10  # it is actually deprecated
__C.TRAIN.learning_rate = 0.0001
__C.TRAIN.learning_rate_minimum = 0.0001
__C.TRAIN.decay_step = 50000
__C.TRAIN.decay_rate = 0.96
__C.TRAIN.max_grad_norm = None

# for the Q learning
__C.TRAIN.value_decay_factor = 0.99

# the number of episodes to play / the number of training step
__C.TRAIN.train_freq = 4

# __C.TRAIN.beta1 = 0.5
# __C.TRAIN.beta2 = 0.999

__C.TRAIN.max_step_size = 100000
__C.TRAIN.max_episode_size = __C.TRAIN.max_step_size / __C.TRAIN.train_freq
__C.TRAIN.snapshot_step = 2000  # save the snapshot every 1000 epoches

# number of episodes to play / number of target network update
__C.TRAIN.update_network_freq = 10000 / __C.TRAIN.train_freq
# when to start the training
__C.TRAIN.training_start_episodes = 50000

# parameters for the exploration
__C.TRAIN.start_epsilon = 1
__C.TRAIN.end_epsilon = 0.01

# parameters for the summary tensorboard
__C.TRAIN.update_tensorboard_episode_length = 0.01

# configurations about the game
__C.GAME.type = 'atari'
__C.GAME.display = False
__C.GAME.screen_size = 80
__C.GAME.history_length = 4
__C.GAME.n_action_repeat = 4
__C.GAME.n_random_action = 30
__C.GAME.max_reward_clip = 1
__C.GAME.return_cumulated_reward = 4

# configuration about the experience shop
__C.EXPERIENCE.size = 1000000  # 1e6
