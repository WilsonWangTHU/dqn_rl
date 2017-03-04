# -----------------------------------------------------------------------------
#   @brief: 
#       We will consider three types of network. The DQN-CNN network, dueling-
#       CNN network, and the actor-critic network (or maybe the a3c, but that
#       should be defined in the agent.py)
#   
#   @author:
#       Tingwu Wang, 1st/Mar/2017
# -----------------------------------------------------------------------------


import tensorflow as tf
from . import cnn
from . import layers
from ..util import logger

class network(object):
    def __init__(self, sess, name, config):

        self.sess = sess  # the tf session
        self.game_type = config.GAME.type.lower()
        self.network_type = config.NETWORK.network_type
        self.network_basebone = config.NETWORK.network_basebone
        self.data_format = config.data_format

        self.config = config
        self.name = name
        self.all_var = {}

        assert self.game_type in ['atari'], \
            logger.error('Game type not supported: {}'.format(self.game_type))
        logger.info('Building network of type: {}, using basebone: {}'.format(
            self.network_type, self.network_basebone))

        self.build_baseline()
        return

    def build_baseline(self):
        self.baseline_net = cnn.basebone_network(
            self.config, self.game_type, self.network_basebone)

    def build_model(self):
        raise NotImplementedError()

    def set_all_var(self):
        self.all_var.update(self.baseline_net.get_all_var())
        return
    def set_all_var_copy_op(self, target_network_var):
        # get the variables in the baseline network
        copy_op = []

        for var in self.all_var:
            copy_op.append(var.assign(target_network_var[var.name]))

        self.copy_op = tf.group(*copy_op, name='copy_to_target_network')

        return
        
    def run_copy(self):
        self.sess.run(self.copy_op)
        return

    def get_var_dict(self):
        return self.all_var
    return

class deep_Q_network(network):
    def __init__(self, sess, name, config, action_space_size):
        # init the parent class and build the baseline model
        with tf.varaible_scope(name):
            super(self.__class__, self).__init__(sess, name, config)
        
            # build the model and record all the trainable variables
            self.build_model()
            self.action_space_size = action_space_size

            # build the whole model
            self.set_all_var()
        return

    def build_model(self):
        self.intermediate_layer = self.baseline_net.get_output_layer()

        # size [b, number_action], no relu, we get the logit
        self.output, self.all_var['w_out'], self.all_var['b_out'] = \
            layers.linear(self.intermediate_layer, self.action_space_size,
                          data_format=self.data_format, name='output')

        self.action = tf.argmax(self.output, dimension=1)
        return
    return

class actor_critic_network(network):
    def __init__(self):
        raise NotImplementedError()
    return

class dueling_network(network):
    def __init__(self):
        raise NotImplementedError()
    return
