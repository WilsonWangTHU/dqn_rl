# -----------------------------------------------------------------------------
#   @brief:
#       The structure that keeps all the history experience
#   @author:
#       Tingwu Wang
# -----------------------------------------------------------------------------

import numpy as np
import random
from ..util import logger

class experience_shop(object):
    def __init__(self, history_length, memory_length, screen_size, batch_size):
        logger.info('building the experience shop')

        self.history_length = history_length
        self.memory_length = memory_length
        self.screen_size = screen_size
        self.batch_size = batch_size

        # the FIFO pointer
        self.current = 0
        self.count = 0

        # locate the memory
        self.action = np.empty(self.memory_length, dtype=np.uint8)
        self.rewards = np.empty(self.memory_length, dtype=np.int8)
        self.observation = np.empty(
            [self.memory_length, self.screen_size, self.screen_size],
            dtype=np.uint8)
        self.terminal = np.empty(self.memory_length, dtype=np.bool)
        
        # the actual output
        self.start_state = np.empty(
            [self.batch_size, self.history_length,
                self.screen_size, self.screen_size])

        self.end_state = np.empty(
            [self.batch_size, self.history_length,
                self.screen_size, self.screen_size])

        return

    def save(self):
        return

    def restore(self):
        return

    def push(self, action, rewards, observation, terminal):
        # record the data
        self.action[self.current] = action
        self.rewards[self.current] = rewards
        self.observation[self.current] = observation
        self.terminal[self.current] = terminal

        # update parameters
        self.current = (self.current + 1) % self.memory_length
        self.count = max(self.count + 1, self.memory_length)

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
                if self.terminal[index - self.history_length: index].any():
                    continue

                # test past, use this index as a training sample
                batch_id.append(index)
                self.start_state[i_batch, ...] = \
                    self.observations[index - self.history_length: index, ...]

                self.end_state[i_batch, ...] = \
                    self.observations[index - self.history_length + 1:
                                      index + 1, ...]

        return self.start_state, self.action[batch_id], \
            self.rewards[batch_id], self.end_state, self.terminal[batch_id]

class history_recorder(object):
    def __init__(self, history_length, screen_size):
        self.size = screen_size
        self.history_length = history_length
        self.history_exp = np.empty(
            [history_length, screen_size, screen_size], np.uint8)

    def update_history(self, new_observe):
        self.history_exp[1:, :, :] = \
            self.history_exp[0: self.history_length, :, :]
        self.history_exp[0, :, :] = new_observe
        return

    def get_history(self):
        return self.history_exp

    def clean_history(self):
        self.history_exp = np.empty(
            [self.history_length, self.size, self.size], np.uint8)
    return
