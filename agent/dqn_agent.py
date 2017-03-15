# -----------------------------------------------------------------------------
#   @brief:
#       define the agents of dqn and dueling network
#   @author:
#       Tingwu Wang
# -----------------------------------------------------------------------------

import init_path
from util import logger
from environment.environments import atari_environment
from model.network import deep_Q_network
from experience import experience_shop, history_recorder
from summary_handler import gym_summary_handler
import tensorflow as tf
import random
import os


class qlearning_agent(object):

    def __init__(self, sess, config, env_name, restore_path=None):
        '''
            @brief:
                The agent has several components:
                @self.env:
                    It is the structure that keep the game environments.
                @self.target_net:
                    The network that's responsible for producing experiences,
                    and the network that changes slowly
                @self.predict_net:
                    The network that's responsible for training and this
                    network changes quickly
                @self.experience:
                    The structure used to record past experience
                @self.summary_handle:
                    The structure that handles all the summary recording

            @what it should do:
                It is a master mind of different functions. But details of
                different function should not appear here (say training,
                storing experience, record summary)
        '''
        self.config = config
        self.env_name = env_name
        self.sess = sess
        logger.info(
            'Constructing a q-learning agent to play {}'.format(env_name))

        # self.step is just a shadow of the self.step_count in the
        # self.predict_net
        self.step = 0

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
        self.target_network = deep_Q_network(
            self.sess, 'target_network', self.config, self.n_action)
        self.predict_network = deep_Q_network(
            self.sess, 'predict_network', self.config, self.n_action,
            target_network=self.target_network,
            update_freq=self.config.TRAIN.update_tensorboard_episode_length)

        # construct the operation of copying weights.
        self.target_network.set_all_var_copy_op(
            self.predict_network.get_var_dict())

        # construct the experience data struct, and a small history shop
        self.exp_shop = experience_shop(self.config.GAME.history_length,
                                        self.config.EXPERIENCE.size,
                                        self.config.GAME.screen_size,
                                        self.config.TRAIN.batch_size)
        self.history_recorder = history_recorder(
            self.config.GAME.history_length, self.config.GAME.screen_size)

        # construct the summary recorder
        self.summary_handler = gym_summary_handler(
            self.sess, self.predict_network.get_td_loss(),
            self.config.TRAIN.update_tensorboard_episode_length)

        # init the network saver and restore
        self.saver = tf.train.Saver()
        if restore_path is not None:
            self.restore_all(restore_path)
        else:
            self.sess.run(tf.global_variables_initializer())
            self.step = 0
        return

    def save_all(self):
        '''
            @brief:
                save all the network parameters and experiences
        '''
        base_path = init_path.get_base_dir()
        path = os.path.join(base_path,
                            'checkpoint', 'dqn_' + str(self.step) + '.ckpt')
        self.saver.save(self.sess, path)

        logger.info('checkpoint saved to {}'.format(path))
        # save the experience shop
        self.exp_shop.save(path)
        return

    def restore_all(self, restore_path):
        '''
            @brief:
                restore the network, the experience shop and the history
                recorder
        '''
        # restore the tf network
        self.saver.restore(self.sess, restore_path)
        logger.info('checkpoint restored from {}'.format(restore_path))
        logger.info('continues from step {}'.format(self.step))
        self.step = self.predict_network.get_step_count()

        # restore the experience shop
        self.exp_shop.restore(restore_path)
        return

    def train_process(self):
        '''
            @brief:
                for the network training, there is two important steps:
                1. generating the experiences
                2. using these experiences to update the network
                3. transfer knowledge from predict network to target network
        '''
        # TODO: make sure the two network are indentical if ...
        while True:
            # generating played sequences
            self.generate_experience()

            # train the network
            self.train_step()

            # save the network if needed
            '''
            if self.step % self.config.TRAIN.snapshot_step == 1:
                self.save_all()
            '''

            # save the played video if needed
            if self.step % self.config.TRAIN.play_and_save_video == 1:
                self.play_game_and_save()
        return

    def play_game_and_save(self):
        # save the video and play a little bit
        base_path = init_path.get_base_dir()
        path = os.path.join(base_path, 'video',
                            init_path.get_time() + 'dqn_' +
                            str(self.step) + '_' + self.env_name)
        if not os.path.exists(path):
            os.mkdir(path)
        for i_video in range(10):
            self.env.set_monitor(os.path.join(path, str(i_video)))
            self.generate_experience(num_episode=1)
            self.env.unset_monitor()
        return

    def generate_experience(self, num_episode=None):
        if num_episode is None:
            num_episode = self.config.TRAIN.exp_train_ratio
        for i_exp in range(num_episode):
            # play the whole episodes
            observation, reward, terminal = self.env.new_game(
                run_random_action=self.env.get_if_run_random_action())
            self.history_recorder.init_history(observation)
            num_step_in_episode = 0
            total_reward = reward
            while terminal is False:
                # get the predicted action, note we all use training step
                # instead of episode count
                epsilon = self.get_epsilon(
                    self.config.TRAIN.end_epsilon,
                    self.config.TRAIN.start_epsilon,
                    self.config.TRAIN.end_epsilon_episode,
                    self.config.TRAIN.training_start_episode)

                if random.uniform(0, 1) <= epsilon:
                    pred_action = random.randint(0, self.n_action - 1)
                else:
                    feed_dict = {}
                    feed_dict[self.predict_network.get_input_placeholder()] = \
                        self.history_recorder.get_history()
                    pred_action = self.predict_network.get_pred_action(
                        feed_dict)

                # do the action and record it in the experience shop
                observation, reward, terminal, _ = self.env.step(pred_action)
                self.exp_shop.push(pred_action, reward, observation, terminal)

                # record it in the history recorder
                self.history_recorder.update_history(observation)

                # record the para to be written into the summary writer
                num_step_in_episode += 1
                total_reward += reward

            # show some information
            if self.exp_shop.episode % 1000 == 0:
                logger.info('episode: {}, epsilon: {}'.format(
                    self.exp_shop.episode, epsilon))
            # add it to the summary handler
            self.summary_handler.add_stat(total_reward, num_step_in_episode,
                                          self.exp_shop.episode)

        return

    def train_step(self):
        if self.exp_shop.count < self.config.TRAIN.training_start_episode:
            return
        else:
            # update the target network if needed. by doing this, we also make
            # sure that the original values in two networks are the same
            if self.step % self.config.TRAIN.update_network_freq == 0:
                self.target_network.run_copy()
                logger.info('At time step {},'.format(self.step) +
                            ' the target_network is updated')

            # fetch the data and train the network
            start_states, end_states, actions, rewards, terminal = \
                self.exp_shop.pop()
            self.step, td_loss = self.predict_network.train_step(
                start_states, end_states, actions, rewards, terminal,
                self.summary_handler.get_td_loss_summary())

            # record the summary of loss
            self.summary_handler.train_writer.add_summary(
                td_loss, self.step)
        return

    def get_epsilon(self, ep_end, ep_start, t_ep_end, t_learn_start):
        current_episode = self.exp_shop.episode
        effective_length = t_ep_end - max(0., current_episode - t_learn_start)
        epsilon = ep_end + \
            max(0., (ep_start - ep_end) * effective_length / float(t_ep_end))
        return epsilon
