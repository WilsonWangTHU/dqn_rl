# -----------------------------------------------------------------------------
#   @brief:
#       In this file, we define all the parameters that relate to the netwrok
#   @author:
#       Tingwu Wang
# -----------------------------------------------------------------------------

from easydict import EasyDict as edict

__NET = edict()

network_config = __NET

__NET.network_basebone = 'nips'  # ['nips', 'nature'], maybe nlp later
__NET.network_type = 'dqn'  # ['dqn', 'duel', 'actor-critic']
__NET.data_format = 'NCHW'  # 'NCHW' is more efficient in cudnn
