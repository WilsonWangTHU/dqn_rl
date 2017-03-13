# ------------------------------------------------------------------------------
#   @brief:
#       In this function, we define a good environment interface for the atari.
#       It's worth mentioning that the we are just sort of add one extra layer
#   @author:
#       Tingwu Wang
# ------------------------------------------------------------------------------

import __init_path
from util import logger
from scipy.misc import imresize
import gym
import random
import numpy as np
__init_path.bypass_frost_warning()


class game_environment(object):

    def __init__(self, env_name, n_action_repeat,
                 display, data_format='NCHW', return_cumulated_reward=False,
                 is_training=True):
        '''
            @brief: init the environment
            @input:
                @env_name: the name of the game
                @n_action_repeat: the number of same action to be excecuted
                @data_format: 'NWHC' or 'NCWH' useful when defining the network
        '''
        self.name = env_name
        self.n_action_repeat = n_action_repeat
        self.env = gym.make(env_name)
        self.data_format = data_format
        self.display = display
        self.return_cumulated_reward = return_cumulated_reward
        self.is_train = is_training

        if self.data_format != 'NCHW':
            logger.warning('You sure you want to use NWHC format?')

        assert n_action_repeat >= 1, \
            logger.error('Action must be at least used once')
        logger.info('Init game environments {}'.format(self.name))

    def new_game(self, n_random_action=0):
        raise NotImplementedError()

    def step(self):
        raise NotImplementedError()


class atari_environment(game_environment):
    '''
        @brief:
            for the atari game, action space is discrete
    '''

    def __init__(self, env_name, n_action_repeat,
                 n_random_action, screen_size,
                 display, data_format='NCHW', return_cumulated_reward=False,
                 is_training=True):

        # init the base environment class
        super(self.__class__, self).__init__(
            env_name, n_action_repeat, display,
            data_format, return_cumulated_reward, is_training)

        self.n_random_action = n_random_action
        if self.n_random_action > 0:
            self.run_random_action = True
        self.screen_size = screen_size

        logger.info('Game set image size: {}, random walk step: {}'.format(
            self.screen_size, self.n_random_action))

        # TODO: display not implemented
        if self.display:
            logger.error('Rendering not implemented / working!')
        return

    def new_game(self, run_random_action=False):
        '''
            @brief:
                start a new game.
            @input:
                @n_random_action: if not 0, than we do some 'noop' action
        '''
        screen = self.env.reset()  # set for new game
        screen, reward, terminal, _ = self.env.step(0)  # the noop action

        # if run random for some steps
        if run_random_action:
            # @TODO: about early terminaled games?
            for i_random_walk in range(random.randint(0, self.n_random_action)):
                screen, reward, terminal, _ = self.env.step(0)  # noop action
                if terminal:  # set for a new game if terminated
                    screen = self.env.reset()
                    screen, reward, terminal, _ = self.env.step(0)
                    logger.warning('New game terminated after {} step'.format(
                        i_random_walk))
        # rendering
        if self.display:
            self.env.render()

        # now return the true observation, reward, terminal, ...
        self.lives = self.env.ale.lives()
        terminal = False
        reward = 0
        observation = self.get_observation(screen)

        return observation, reward, terminal

    def step(self, action):
        assert self.lives >= 0, logger.error(
            'Why negative lives? did you run new_game()?')
        cumulated_reward = 0
        screen, reward, terminal, _ = self.env.step(action)

        for _ in range(self.n_action_repeat):
            screen, reward, terminal, _ = self.env.step(action)
            cumulated_reward += reward
            current_lives = self.env.ale.lives()

            # in training, even dead by once, we regard as an end of game
            if terminal or (self.is_train and self.lives > current_lives):
                terminal = True
                break

            self.lives = self.env.ale.lives()

        # rendering
        if self.display:
            self.env.render()

        # get what is needed to be returned
        if not terminal:
            self.lives = current_lives
        else:  # to make sure that no one run game without init
            self.lives = -1
        if self.return_cumulated_reward:
            reward = cumulated_reward
        screen = self.get_observation(screen)

        return screen, reward, terminal, {}

    def get_observation(self, screen):
        '''
            @brief:
                from the import raw pixel observation to the real observation
        '''
        screen = screen[:, :, 0] * 0.2126 + screen[:, :, 1] * 0.7152 + \
            screen[:, :, 2] * 0.0722
        screen = screen.astype(np.uint8)
        screen = imresize(screen, [self.screen_size, self.screen_size])

        return screen

    def get_action_space_size(self):
        '''
            @brief: the network will need to know the size of action
        '''
        return self.env.action_space.n

    def get_if_run_random_action(self):
        return self.run_random_action
