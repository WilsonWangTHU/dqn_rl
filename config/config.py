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
__C.TRAIN.learning_rate = 0.00025  # 0.001 (suggested)
__C.TRAIN.learning_rate_minimum = 0.00025
__C.TRAIN.decay_step = 50000
__C.TRAIN.decay_rate = 0.99
__C.TRAIN.max_grad_norm = None

# for the Q learning
__C.TRAIN.value_decay_factor = 0.99

# the number of episodes to play / the number of training step
__C.TRAIN.exp_train_ratio = 4

__C.TRAIN.snapshot_step = 20000 / __C.TRAIN.exp_train_ratio  # 200000

# number of steps to update target network
__C.TRAIN.update_network_freq = 10000 / __C.TRAIN.exp_train_ratio
__C.TRAIN.play_and_save_video = 10000 / __C.TRAIN.exp_train_ratio  # 100000

# parameters for the exploration
__C.TRAIN.start_epsilon = 1
__C.TRAIN.end_epsilon = 0.1
# when to start the training
__C.TRAIN.training_start_episode = 50000  # it is step actually
__C.TRAIN.end_epsilon_episode = 1000000  # it is step actually

# parameters for the summary tensorboard
__C.TRAIN.update_tensorboard_step_length = 100

# configurations about the game
__C.GAME.type = 'atari'
__C.GAME.display = False
__C.GAME.screen_size = 84
__C.GAME.history_length = 4
__C.GAME.n_action_repeat = 1  # TODO: they already have the skip of frame
__C.GAME.n_random_action = 30
__C.GAME.max_reward_clip = 1
__C.GAME.return_cumulated_reward = 4

# configuration about the experience shop
__C.EXPERIENCE.size = 1000000  # 1e6
