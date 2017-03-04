# -----------------------------------------------------------------------------
#   @brief:
#       define the agents of dqn and dueling network
#   @author:
#       Tingwu Wang
# -----------------------------------------------------------------------------

from ..util import logger
from ..environment.environments import atari_environment
from ..model.network import deep_Q_network
from .experience import experience_shop, history_recorder
import tensorflow as tf

class qlearning_agent(object):
    def __init__(self, sess, config, env_name):
        '''
            @brief:
                The agent has several components:
                @self.env:
                    It is the structure that keep the game environments.
                @self.target_net:
                    The network that changes slowly
                @self.predict_net:
                    The network that changes quickly
                @self.experience:
                    The structure used to record past experience
                @self.summary_handle:
                    The structure that handles all the summary recording
                
            @what it should do
                define how to train the network, train the network.
        '''
        self.config = config
        self.env_name = env_name
        self.sess = sess
        logger.info(
            'Constructing a q-learning agent to play {}'.format(env_name))
        self.step = 0  # steps are useful for the training process
        self.episode = 0  # episodes are useful for the generating process
        
        # construct the environment
        if self.config.GAME.type == 'atari':
            self.env = atari_environment(
                self.env_name, self.config.GAME.n_action_repeat,
                self.config.GAME.n_random_action,
                self.config.GAME.screen_size, self.config.GAME.display,
                self.config.NETWORK.data_format,
                self.config.GAME.return_cumulated_reward)
        else:
            assert False, logger.error('Not implemented environment {}'.format(
                self.config.GAME.type))
        self.n_action = self.env.get_action_space_size()

        # construct the networks
        self.target_network = deep_Q_network(self.sess, 'target_network',
                                             self.config, self.n_action)
        self.predict_network = deep_Q_network(self.sess, 'predict_network',
                                              self.config, self.n_action)
        self.target_network.set_all_var_copy_op(
            self.predict_network.get_var_dict())

        # construct the experience data struct, and a small history shop
        self.exp_shop = experience_shop(self.config.GAME.history_length,
                                        self.config.EXPERIENCE.size,
                                        self.config.GAME.screen_size,
                                        self.config.TRAIN.batch_size)
        self.history_recorder = history_recorder(
            self.config.GAME.history_length, self.config.GAME.screen_size)
        return

    def save_all(self):
        # save all the network parameters and experiences
        raise NotImplementedError()
        return

    def restore_all(path, syn_target_net):
        # restore the network and the experiences
        raise NotImplementedError()
        return

    def train_process(self):
        '''
            @brief:
                for the network training, there is two important steps:
                1. generating the experiences
                2. using these experiences to update the network
                3. transfer knowledge from predict network to target network
        '''


        return

    def init_training(self, restore_path=None):
        tf.global_variables_initializer()
        if restore_path is not None:
            self.restore_all(restore_path)
        else:
            self.target_network.run_copy()  # make two network identical

        # run the game and init history
        self.history

    def generate_experience(self):
        return

    def train_network(self):
        return
