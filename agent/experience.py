# -----------------------------------------------------------------------------
#   @brief:
#       The structure that keeps all the history experience
#   @author:
#       Tingwu Wang
# -----------------------------------------------------------------------------

import init_path
import numpy as np
import random
from util import logger

init_path.bypass_frost_warning()


class experience_shop(object):
    '''
        @brief:
            everything you need about the experience
    '''

    def __init__(self, history_length, memory_length, screen_size, batch_size):
        logger.info('building the experience shop')

        self.history_length = history_length
        self.memory_length = memory_length
        self.screen_size = screen_size
        self.batch_size = batch_size

        # the FIFO pointer
        self.current = 0
        self.count = 0
        self.total_episode = 0

        # locate the memory
        self.actions = np.empty(self.memory_length, dtype=np.uint8)
        self.rewards = np.empty(self.memory_length, dtype=np.int8)
        self.observations = np.empty(
            [self.memory_length, self.screen_size, self.screen_size],
            dtype=np.uint8)
        self.terminals = np.empty(self.memory_length, dtype=np.bool)

        # the actual output
        self.start_states = np.empty(
            [self.batch_size, self.history_length,
                self.screen_size, self.screen_size])

        self.end_states = np.empty(
            [self.batch_size, self.history_length,
                self.screen_size, self.screen_size])

        return

    def save(self, path):
        path = path.replace('.ckpt', '.npy')
        data_dict = {'count': self.count,
                     'actions': self.actions[:self.count],
                     'rewards': self.rewards[:self.count],
                     'observations': self.observations[:self.count, :, :],
                     'terminals': self.terminals[:self.count],
                     'current': self.current,
                     'total_episode': self.total_episode}

        np.save(path, data_dict)
        logger.info('   Experience shop saved to {}'.format(path))
        return

    def restore(self, path):
        path = path.replace('.ckpt', '.npy')
        data_dict = np.load(path)
        data_dict = data_dict[()]

        logger.info('Experience shop restored from {}'.format(path))
        self.count = data_dict['count']
        logger.info('{} data in the experience shop now'.format(self.count))

        # restore the data
        self.actions[:self.count] = data_dict['actions']
        self.rewards[:self.count] = data_dict['rewards']
        self.observations[:self.count, :, :] = data_dict['observations']
        self.terminals[:self.count] = data_dict['terminals']
        self.current = data_dict['current']
        self.total_episode = data_dict['total_episode']
        return

    def push(self, action, rewards, observation, terminal):
        # record the data
        self.actions[self.current] = action
        self.rewards[self.current] = rewards
        self.observations[self.current] = observation
        self.terminals[self.current] = terminal

        # update parameters
        self.current = (self.current + 1) % self.memory_length
        self.count = min(self.count + 1, self.memory_length)
        self.total_episode += 1

        return

    def pop(self):
        # TODO: actually there is one bug about this program ...
        # plays that goes between the memory size will not be used
        batch_id = []
        for i_batch in range(self.batch_size):
            # find length = self.history_length frames, that don't have
            # terminal signals in between, loop until found
            while True:
                index = random.randint(self.history_length, self.count - 1)

                # test if it is a valid state: is_broken_experience
                if (index >= self.current) and \
                        (index - self.history_length < self.current):
                    continue
                # test if it is a valid state: is_multiple_play
                if self.terminals[index - self.history_length: index].any():
                    continue

                # test past, use this index as a training sample
                batch_id.append(index)
                self.start_states[i_batch, ...] = \
                    self.observations[index - self.history_length: index, ...]

                self.end_states[i_batch, ...] = \
                    self.observations[index - self.history_length + 1:
                                      index + 1, ...]
                # move on to the next point
                break

        return self.start_states, self.end_states, self.actions[batch_id], \
            self.rewards[batch_id], self.terminals[batch_id]

    @property
    def episode(self):
        return self.total_episode


class history_recorder(object):

    def __init__(self, history_length, screen_size):
        self.size = screen_size
        self.history_length = history_length
        self.history_exp = np.empty(
            [history_length, screen_size, screen_size], np.uint8)

    def update_history(self, observation):
        self.history_exp[1:, :, :] = \
            self.history_exp[0: self.history_length, :, :]
        self.history_exp[0, :, :] = observation
        return

    def init_history(self, observation):
        for i in range(self.history_length):
            self.history_exp[i, :, :] = observation
        return

    def get_history(self):
        '''
            @return (batch_size, history_length, screen_size, screen_size)
        '''
        return np.expand_dims(self.history_exp, axis=0)

    def clean_history(self):
        self.history_exp = np.empty(
            [self.history_length, self.size, self.size], np.uint8)
