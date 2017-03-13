# -----------------------------------------------------------------------------
#   @brief:
#       In this file, we define some basebone network for the rl-network
# -----------------------------------------------------------------------------

import init_path
import tensorflow as tf
from util import logger
import layers
init_path.bypass_frost_warning()


class basebone_network(object):
    '''
        @brief:
            currently we support nips type and nature type
    '''

    def __init__(self, config, game_type, network_basebone):
        logger.info('building the basebone network, using {} format'.format(
            network_basebone))

        # make the reference chain shorter
        self.game_type = game_type  # this might be important later
        self.network_basebone = network_basebone

        self.batch_size = None
        self.screen_size = config.GAME.screen_size
        self.history_length = config.GAME.history_length
        self.data_format = config.NETWORK.data_format

        self.config = config.NETWORK

        # record the variables
        self.var = {}

        # now build the network
        self.build_model()
        return

    def build_model(self):
        # the input placeholder
        if self.data_format == 'NCHW':
            self.input_screen = tf.placeholder(
                'float32',
                [self.batch_size, self.history_length,
                    self.screen_size, self.screen_size])
        else:
            self.input_screen = tf.placeholder(
                'float32',
                [self.batch_size, self.screen_size,
                    self.screen_size, self.history_length])
            logger.warning(
                'data_format |{}| might not be a good idea, you sure?')

        # the convolutional layers
        self.l0 = self.input_screen / 255.0  # size [b, 4, 80, 80]

        if self.network_basebone == 'nature':
            # self.l1: size [b, 32, 20, 20], relu used
            self.l1, self.var['l1_w'], self.var['l1_h'] = layers.conv2d(
                self.l0, 32, [8, 8], [4, 4], data_format=self.data_format,
                name='conv1')

            # self.l2: size [b, 64, 10, 10], relu used
            self.l2, self.var['l2_w'], self.var['l2_h'] = layers.conv2d(
                self.l1, 64, [4, 4], [2, 2], data_format=self.data_format,
                name='conv2')

            # self.l3: size [b, 64, 10, 10], relu used
            self.l3, self.var['l3_w'], self.var['l3_h'] = layers.conv2d(
                self.l2, 64, [3, 3], [1, 1], data_format=self.data_format,
                rame='conv3')

            # self.l4: size [b, 512], relu used
            self.l4, self.var['l4_w'], self.var['l4_h'] = layers.linear(
                self.l3, 512, activation_fn=tf.nn.relu, name='fc_4')

            self.output = self.l4
        elif self.network_basebone == 'nips':
            # self.l1: size [b, 16, 20, 20], relu used
            self.l1, self.var['l1_w'], self.var['l1_h'] = layers.conv2d(
                self.l0, 16, [8, 8], [4, 4], data_format=self.data_format,
                name='conv1')

            # self.l2: size [b, 32, 10, 10], relu used
            self.l2, self.var['l2_w'], self.var['l2_h'] = layers.conv2d(
                self.l1, 32, [4, 4], [2, 2], data_format=self.data_format,
                name='conv2')

            # self.l3: size [b, 512], relu used
            self.l3, self.var['l3_w'], self.var['l3_h'] = layers.linear(
                self.l2, 256, activation_fn=tf.nn.relu, name='fc_3')

            self.output = self.l3
        else:
            assert False, logger.error(
                'Unknown baseline type {}'.format(self.network_basebone))

        return

    def get_all_var(self):
        return self.var

    def get_output_layer(self):
        return self.output

    def get_input_placeholder(self):
        return self.input_screen
