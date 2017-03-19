# -----------------------------------------------------------------------------
#   @brief:
#       We train the gym player in this part.
#   @author:
#       Tingwu Wang, 21st, Feb, 2017
#   @TODO:
#       1. Change the feed-dict into multi-thread-fashion
# -----------------------------------------------------------------------------

import init_path
from util import logger
from agent import dqn_agent
from config.config import base_config as config
import tensorflow as tf
import argparse
from environment import corridor
init_path.bypass_frost_warning()


def change_debug_config(config):
    '''
        --network_header_type=mlp
        --observation_dims='[16]'
        --env_name=CorridorSmall-v5
        --t_learn_start=0.1
        --learning_rate_decay_step=0.1
        --history_length=1
        --n_action_repeat=1
        --t_ep_end=10
        --display=True
        --learning_rate=0.025
        --learning_rate_minimum=0.0025
    '''
    config.NETWORK.network_basebone = 'mlp'
    config.GAME.screen_size = 4
    config.TRAIN.training_start_episode = 1000
    config.TRAIN.end_epsilon_episode = 20000
    config.TRAIN.decay_step = 1000
    config.GAME.history_length = 1
    config.TRAIN.learning_rate = 0.005
    config.TRAIN.learning_rate_minimum = 0.005
    config.TRAIN.update_network_freq = 2000


    # the environment
    config.GAME.type = 'toy'

    return config


if __name__ == '__main__':
    # the parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default=0)
    parser.add_argument('--restore', help='the path of model to restore',
                        default=None)
    parser.add_argument('--env_name',
                        help='the game to play, add the deterministic flag',
                        default='Breakout-v0')

    parser.add_argument('--debug',
                        help='the game to play, add the deterministic flag',
                        default=False)

    args = parser.parse_args()
    args.debug = True
    config.TRAIN.training_start_episode = 1000
    # init the logger, just save the network
    logger.set_file_handler(prefix='gym_')

    # if debug, make some changes to the config file
    if args.debug:
        config = change_debug_config(config)
        args.env_name = 'CorridorSmall-v5'

    # build the network
    sess = tf.Session()
    tf.device('/gpu:' + str(args.gpu))
    logger.info('Session starts, using gpu: {}'.format(str(args.gpu)))
    game_agent = dqn_agent.qlearning_agent(
        sess, config, args.env_name, restore_path=args.restore)

    # train the network
    game_agent.train_process()
