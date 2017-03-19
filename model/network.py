# -----------------------------------------------------------------------------
#   @brief:
#       We will consider three types of network. The DQN-CNN network, dueling-
#       CNN network, and the actor-critic network (or maybe the a3c, but that
#       should be defined in the agent.py)
#
#   @author:
#       Tingwu Wang, 1st/Mar/2017
#   @possible compatible issues:
#       tf.max is no longer valid after tf 1.0, it's transformed into
#       tf.reduce_max
# -----------------------------------------------------------------------------

import init_path
import tensorflow as tf
import cnn
import layers
from util import logger
import numpy as np
from matplotlib import pyplot as plt
init_path.bypass_frost_warning()


class network(object):

    def __init__(self, sess, name, config):
        '''
            @brief:
                Define the network for rl prediction
            @input:
                @self.network_basebone:
                    The basebone of the network is the most important part,
                    which defines what the network's structure is no matter
                    what different deep rl algorithm it wants to use.
                @self.network_type:
                    At the same time, this variable define the different types
                    of the network output (dueling, normal dqn, or double Q)
        '''
        self.sess = sess  # the tf session
        self.game_type = config.GAME.type.lower()
        self.network_type = config.NETWORK.network_type
        self.network_basebone = config.NETWORK.network_basebone
        self.data_format = config.NETWORK.data_format
        self.output = None  # this will be defined later

        self.config = config  # just store all the information
        self.name = name
        self.all_var = {}

        logger.info('Building network of type: {}, using basebone: {}'.format(
            self.network_type, self.network_basebone))

        self.build_baseline()
        return

    def build_baseline(self):
        self.baseline_net = cnn.basebone_network(
            self.config, self.game_type, self.network_basebone)
        self.input_screen = self.baseline_net.get_input_placeholder()

    def build_model(self):
        raise NotImplementedError()

    def set_all_var(self):
        # get the parameters in the baseline network recorded
        assert len(self.all_var) > 0, logger.error(
            'Wait? Where are the output parameters?')
        baseline_var = self.baseline_net.get_all_var()

        for var_name in baseline_var:
            self.all_var['base_' + var_name] = baseline_var[var_name]
        return

    def set_all_var_copy_op(self, prediction_network_var):
        # get the variables in the baseline network
        copy_op = []

        for var_name in self.all_var:
            copy_op.append(self.all_var[var_name].assign(
                prediction_network_var[var_name]))

        self.copy_op = tf.group(*copy_op, name='copy_to_target_network')
        return

    def run_copy(self):
        self.sess.run(self.copy_op)
        return

    def get_var_dict(self):
        '''
            @brief: it is going to be called by the prediction network
        '''
        return self.all_var

    def get_pred_action(self, feed_dict):
        # TODO: we might want to know how to deal with asyn
        return self.sess.run(self.action, feed_dict=feed_dict)[0]

    def get_input_placeholder(self):
        return self.baseline_net.get_input_placeholder()


class deep_Q_network(network):
    '''
        @brief:
            In this part, we define the deep_Q_network. Note that both the
            prediction network and the target network is defined here.
            Their only difference will be the "target_network" input

        @note:
            The prediction_network is responsible to train the network, so it
            will need the reference of the target_network to generate
            experiences
    '''

    def __init__(self, sess, name, config, action_space_size,
                 target_network=None, update_freq=100):
        # init the parent class and build the baseline model
        self.target_network = target_network
        self.info_update_freq = update_freq

        if target_network is None:
            logger.info('building the target network, name: {}'.format(name))
        else:
            logger.info('building the pred network, name: {}'.format(name))

        with tf.variable_scope(name):
            super(self.__class__, self).__init__(sess, name, config)

            # build the model and record all the trainable variables
            self.action_space_size = action_space_size
            self.build_model()

            # build the whole model
            self.set_all_var()
        return

    def build_model(self):
        self.intermediate_layer = self.baseline_net.get_output_layer()

        # size [batch_size, num_action_space], no relu, we get the logit
        self.output, self.all_var['w_out'], self.all_var['b_out'] = \
            layers.linear(self.intermediate_layer, self.action_space_size,
                          name='output')

        self.action = tf.argmax(self.output, dimension=1)

        self.max_Q_value = tf.reduce_max(self.output, axis=1)

        # if it is the target network, then we could just return
        if self.target_network is None:
            return

        # for the prediction network, we need to do something else
        self.Q_next_state_input = tf.placeholder('float32', [None])
        self.reward_input = tf.placeholder('float32', [None])
        self.action_input = tf.placeholder('int64', [None])

        # size [batch_size, num_action_space]
        self.actions_one_hot = tf.one_hot(self.action_input,
                                          self.action_space_size,
                                          1.0, 0.0, name='action_one_hot')
        # size [batch_size]
        self.pred_state_action_value = tf.reduce_sum(
            self.output * self.actions_one_hot,
            reduction_indices=1, name='action_state_value')
        
        # size [batch_size]
        self.target_value = self.config.TRAIN.value_decay_factor * \
            self.Q_next_state_input + self.reward_input
        
        self.td_diff = self.target_value - self.pred_state_action_value
        
        self.td_diff_clipped = tf.where(tf.abs(self.td_diff) < 1.0,
                                        0.5 * tf.square(self.td_diff),
                                        tf.abs(self.td_diff) - 0.5,
                                        name='clipped_loss')
        self.td_loss = tf.reduce_mean(self.td_diff_clipped, name='final_loss')

        self.init_training()
        #self.action_grads = tf.gradients(self.td_loss, self.baseline_net.layers['l_2'])
        
    def get_next_state_value(self, end_states, terminal):
        assert self.target_network is None, logger.error(
            'You should call the target network to generate the value')
        max_value = self.sess.run(
            self.max_Q_value, feed_dict={self.input_screen: end_states})
        max_value[np.where(terminal)] = 0  # if terminated, return 0
        return max_value

    def train_step(self, start_states, end_states, actions, rewards, terminal,
                   td_loss_summary):
        assert self.target_network is not None, logger.error(
            'You should call the prediction network')

        # get the target prediction
        Q_next_state = self.target_network.get_next_state_value(end_states,
                                                                terminal)

        # increment the step parameters
        self.sess.run(self.step_add_op)

        current_step = self.sess.run(self.step_count)
        if current_step < -1 and current_step % 10000 == 1:            
            current_loss, current_step, td_loss_sum, q_values, diff_clipped = self.sess.run(
                [self.td_loss, self.step_count, td_loss_summary, 
                 self.output, self.td_diff_clipped],
                feed_dict={self.action_input: actions,
                           self.reward_input: rewards,
                           self.Q_next_state_input: Q_next_state,
                           self.input_screen: start_states})

            for i in range(32):
                self.show_result(actions[i], rewards[i], start_states[i], 
                                 end_states[i], terminal[i], Q_next_state[i], q_values[i], diff_clipped[i])
        for _ in range(20):
            _, current_loss, current_step, td_loss_sum, q_values = self.sess.run(
                [self.optim, self.td_loss, self.step_count, td_loss_summary, 
                 self.output],
                feed_dict={self.action_input: actions,
                           self.reward_input: rewards,
                           self.Q_next_state_input: Q_next_state,
                           self.input_screen: start_states})
            #print 'lllloooosssseeeee: {}'.format(current_loss)
            #print '1: {}'.format(q_values[0])
            #print '2: {}'.format(q_values[1])
        
        #print '-----------------------'
        #print q_values[0]
        #print q_values[1]
        #plt.imsave('test.png', np.array(start_states[0,0], dtype=np.float) / 255)
        #print '-----------------------'
        #print self.all_var['base_w_2'].eval(self.sess)
        #print current_loss
        #if current_step % self.info_update_freq == 0:
        if current_step % 5 == 0:
            logger.info('step: {}, current TD loss: {}'.format(
                current_step, current_loss))
        '''
        gradients_and_vars = self.sess.run(self.action_grads, 
            feed_dict={self.action_input: actions,
                       self.reward_input: rewards,
                       self.Q_next_state_input: Q_next_state,
                       self.input_screen: start_states})
        '''
        return current_step, td_loss_sum

    def init_training(self):
        '''
            @brief: here we init the network training procedure
        '''

        # build the timestep variable to use during learning rate decay
        self.step_count = tf.Variable(0, trainable=False, name='step')
        self.step_add_op = self.step_count.assign_add(1)

        self.episode_count = tf.Variable(0, trainable=False, name='episode')
        self.episode_add_op = self.episode_count.assign_add(1)

        self.learning_rate_op = tf.maximum(
            self.config.TRAIN.learning_rate_minimum,
            tf.train.exponential_decay(
                self.config.TRAIN.learning_rate,
                self.step_count,
                self.config.TRAIN.decay_step,
                self.config.TRAIN.decay_rate,
                staircase=True))

        self.optimizer = tf.train.RMSPropOptimizer(
            self.learning_rate_op, momentum=0.95, epsilon=0.01)

        '''
        if self.config.TRAIN.max_grad_norm is not None:
            grads_and_vars = self.optimizer.compute_gradients(self.td_loss)
            for idx, (grad, var) in enumerate(grads_and_vars):
                if grad is not None:
                    grads_and_vars[idx] = \
                        (tf.clip_by_norm(grad, self.max_grad_norm), var)
            self.optim = self.optimizer.apply_gradients(grads_and_vars)
        else:
            self.optim = self.optimizer.minimize(self.td_loss)
            '''
        self.optim = self.optimizer.minimize(self.td_loss)

    def get_step_count(self):
        '''
            @brief:
                tell the agent what time it is
        '''
        return self.sess.run(self.step_count)

    def get_td_loss(self):
        return self.td_loss

    def get_max_q(self):
        return self.max_Q_value
    
    def show_result(self, pred_action, reward, start_state, 
                    observation, terminal, predicted_max_value_next_state, current_q_a_value, diff_clipped):
        he_map = ['Left', 'Down', 'Right', 'Up', 'NOOP']
        if reward > 0:
            print('success!')
        print('-------------------- debug result:')
        print(np.array(start_state).reshape(4,4))
        print('action: {}'.format(he_map[pred_action]))
        print('reward: {}'.format(reward))
        print('terminal: {}'.format(terminal))
        print(np.array(observation).reshape(4,4))
        print('target_should be: {}'.format(0.99 * predicted_max_value_next_state + reward))
        print('predicited: {}'.format(current_q_a_value[pred_action]))
        print('all_value: {}'.format(current_q_a_value))
        return


class actor_critic_network(network):

    def __init__(self):
        raise NotImplementedError()
        return


class dueling_network(network):

    def __init__(self):
        raise NotImplementedError()
        return
