# -----------------------------------------------------------------------------
#   @brief:
#       define the how to record the summary
#   @author:
#       Tingwu Wang
# -----------------------------------------------------------------------------

import init_path
from util import logger
import tensorflow as tf
import os


class summary_handler(object):
    '''
        @brief:
            Tell the handler where to record all the information.
            Normally, we want to record the prediction of value loss, and the
            average reward (maybe learning rate)
    '''

    def __init__(self, sess):
        # the interface we need
        self.summary = None
        self.sess = sess
        self.path = os.path.join(init_path.get_base_dir(), 'checkpoint')
        self.train_writer = tf.summary.FileWriter(self.path, sess.graph)

        logger.info(
            'summary write initialized, writing to {}'.format(self.path))

    def get_tf_summary(self):
        assert self.summary is not None, logger.error(
            'tf summary not defined, call the summary object separately')
        return self.summary


class gym_summary_handler(summary_handler):
    '''
        @brief:
            For the gym part, we need to record the
            1. episode average game length
            2. average reward per episode
            3. loss between difference of estimated value (TD)
    '''

    def __init__(self, sess, td_loss, max_q, update_frequency):
        super(self.__class__, self).__init__(sess)
        self.loss_td_sum = tf.summary.scalar('td_loss', td_loss)
        self.maxq_sum = tf.summary.scalar('max_q', tf.reduce_max(max_q))

        self.sum = tf.summary.merge([self.loss_td_sum, self.maxq_sum])
        self.update_frequency_episode = update_frequency

        self.step_counter_per_episode = 0
        self.reward_per_episode = 0
        # init the reward
        self.reset_stat()
        return

    def manually_add_scalar_summary(self, summary_name, summary_value, x_axis):
        '''
            @brief:
                might be useful to record the average game_length, and average
                reward
            @input:
                x_axis could either be the episode number of step number
        '''
        summary = tf.Summary(
            value=[tf.Summary.Value(
                tag=summary_name, simple_value=summary_value), ])
        self.train_writer.add_summary(summary, x_axis)
        return

    def get_td_loss_summary(self):
        return self.sum

    def add_episode_stat(self, game_step_count):
        # flush the data recorded during one single episode
        self.reward_total += self.reward_per_episode
        self.episode_length_total += self.step_counter_per_episode
        self.count += 1

        # get prepared for the new episode, clean data
        self.reward_per_episode = 0
        self.step_counter_per_episode = 0

        if self.count >= self.update_frequency_episode:
            # record the data into the summary
            self.episode_length_total = \
                self.episode_length_total / float(self.count)
            self.reward_total = self.reward_total / float(self.count)

            self.manually_add_scalar_summary(
                'avg_reward', self.reward_total, game_step_count)

            self.manually_add_scalar_summary(
                'avg_episode_length', self.episode_length_total,
                game_step_count)

            logger.info(
                'Current step: {}, Reward: {} (over {} episodes)'.format(
                    game_step_count, self.reward_total,
                    self.update_frequency_episode))
            logger.info('Length: {}'.format(self.episode_length_total))

            self.reset_stat()
        return

    def reset_stat(self, init_reward=0):
        self.reward_total = init_reward
        self.episode_length_total = 0

        self.count = 0
        return

    def add_step_stat(self, reward):
        self.reward_per_episode += reward
        self.step_counter_per_episode += 1
        return
