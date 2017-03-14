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

    def __init__(self, sess, td_loss, update_frequency):
        super(self.__class__, self).__init__(sess)
        self.loss_td_sum = tf.summary.scalar('td_loss', td_loss)
        self.update_frequency_episode = update_frequency

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
        return self.loss_td_sum

    def add_stat(self, reward, length, episode_count):
        self.reward_total += reward
        self.episode_length_total += length
        self.count += 1
        if self.count > self.update_frequency_episode:
            # record the data into the summary
            self.episode_length_total = \
                self.episode_length_total / float(self.count)
            self.reward_total = self.reward_total / float(self.count)

            self.manually_add_scalar_summary(
                'avg_reward', self.reward_total, episode_count)

            self.manually_add_scalar_summary(
                'avg_episode_length', self.episode_length_total, episode_count)

            logger.info(
                'At episode: {}, Reward: {} (over {} episodes)'.format(
                    episode_count, self.reward_total,
                    self.update_frequency_episode))
            logger.info('Length: {}'.format(self.episode_length_total))

            self.reset_stat()
        return

    def reset_stat(self):
        self.reward_total = 0
        self.episode_length_total = 0

        self.count = 0
        return
