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
init_path.bypass_frost_warning()

if __name__ == '__main__':
    # the parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default=0)
    parser.add_argument('--restore', help='the path of model to restore',
                        default=None)
    parser.add_argument('--env_name', help='the game to play',
                        default='Breakout-v0')

    args = parser.parse_args()

    # init the logger, just save the network
    logger.set_file_handler(prefix='gym_')

    # build the network
    sess = tf.Session()
    tf.device('/gpu:' + str(args.gpu))
    logger.info('Session starts, using gpu: {}'.format(str(args.gpu)))
    game_agent = dqn_agent.qlearning_agent(
        sess, config, args.env_name, restore_path=args.restore)

    # train the network
    game_agent.train_process()
