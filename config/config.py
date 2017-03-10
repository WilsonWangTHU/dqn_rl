# -----------------------------------------------------------------------------
#   @brief:
#       In this file, all the parameters of the network is configurated
#       For different network, we use separate parameters
#
# -----------------------------------------------------------------------------


from easydict import EasyDict as edict
from .network_config import network_config

__C = edict()
base_config = __C

__C.NETWORK = network_config
__C.ATARI = edict()  # se store the 
__C.EXPERIENCE = edict()
__C.TRAIN = edict()
__C.TEST = edict()

# basic training parameters
__C.TRAIN.batch_size = 32
__C.TRAIN.gradient_clip = 10
__C.TRAIN.learning_rate = 0.0001
__C.TRAIN.learning_rate_minimum = 0.0001
__C.TRAIN.beta1 = 0.5
__C.TRAIN.beta2 = 0.999

__C.TRAIN.max_step_size = 100000
__C.TRAIN.snapshot_step = 2000  # save the snapshot every 1000 epoches
# after how many training step, do we update the network
__C.TRAIN.update_network_freq = 1000


# configurations about the game
__C.GAME.type = 'atari'
__C.GAME.display = False
__C.GAME.screen_size = 80
__C.GAME.history_length = 4
__C.GAME.n_action_repeat = 4
__C.GAME.n_random_action = 4
__C.GAME.max_reward_clip = 1
__C.GAME.return_cumulated_reward = 4


# configuration about the experience shop
__C.EXPERIENCE.size = 1000000  # 1e6
# the number of episodes to play / the number of training step
__C.EXPERIENCE.exp_train_ratio = 4
