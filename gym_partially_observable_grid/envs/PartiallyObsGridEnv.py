from copy import deepcopy

import gym
from gym import spaces

from gym_partially_observable_grid.utils import PartiallyObsGridworldParser


class PartiallyObservableWorld(gym.Env):
    def __init__(self,
                 world_file_path,
                 force_determinism=False,
                 indicate_slip=False,
                 is_partially_obs=True,
                 indicate_wall=False,
                 max_ep_len=100,
                 goal_reward=100,
                 one_time_rewards=True,
                 step_penalty=0):

        # Available actions
        self.actions_dict = {'up': 0, 'down': 1, 'left': 2, 'right': 3}
        self.action_space_to_act_map = {i:k for k,i in self.actions_dict.items()}
        self.actions = [0, 1, 2, 3]

        parser = PartiallyObsGridworldParser(world_file_path)

        # State space size from layout file
        self.state_space = parser.state_space
        # Representation of concrete ((x,y) coordinates) world and abstract world
        self.world, self.abstract_world = parser.world, parser.abstract_world
        # Map of abstract symbols to their names (if any)
        self.abstract_symbol_name_map = parser.abstract_symbol_name_map
        # Map of stochastic tiles, where each tile is identified by rule_id
        self.rules = parser.rules
        # Map of locations to rule_ids, that is, tile has stochastic behaviour
        # force_determinism - this option exist if you want to make a stochastic env. deterministic
        self.stochastic_tile = parser.stochastic_tile if not force_determinism else dict()
        # Map of locations that return a reward
        self.reward_tiles = parser.reward_tiles
        # If one_time_rewards set to True, reward for that tile will be receive only once during the episode
        self.one_time_rewards = one_time_rewards
        self.collected_rewards = set()

        # If true, once the executed action is not the same as the desired action,
        # 'slip' will be added to abstract output
        self.indicate_slip = indicate_slip
        self.slip_action = None

        # If indicate_wall is set to True, suffix '_wall' will be added once the agent runs into the wall
        self.indicate_wall = indicate_wall

        # Indicate whether observations will be abstracted or will they be x-y coordinates
        self.is_partially_obs = is_partially_obs

        # If abstraction is not defined, environment cannot be partially observable
        if self.abstract_world is None:
            self.is_partially_obs = False

        # Layout variables
        self.initial_location = parser.initial_location
        self.player_location = parser.player_location
        self.goal_locations = parser.goal_location
        self.terminal_locations = parser.terminal_locations
        self.behavioral_toggles = parser.behavioral_toggles

        # Should stochastic behaviour be enabled
        self.use_stochastic_tiles = True

        # Reward reached when reaching goal or negative amount when terminal state will be reached
        self.goal_reward = goal_reward

        # Step penalty that will be returned every every step if the reward is not reached
        self.step_penalty = step_penalty if step_penalty < 0 else step_penalty * -1

        # Episode length
        self.max_ep_len = max_ep_len
        self.step_counter = 0

        # Action and Observation Space
        self.one_hot_2_state_map, self.one_hot_2_state_map = None, None
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Discrete(self._get_obs_space())

    def _get_obs_space(self):
        self.state_2_one_hot_map = {}
        counter = 0
        abstract_symbols = set()
        world_to_process = self.world if not self.is_partially_obs else self.abstract_world
        for x, row in enumerate(world_to_process):
            for y, tile in enumerate(row):
                if tile not in {'#', 'D', 'E'}:
                    if self.is_partially_obs:
                        if tile == ' ' or tile == 'G':
                            self.state_2_one_hot_map[(x, y)] = counter
                            counter += 1
                        else:
                            abstraction = self.abstract_symbol_name_map[tile]
                            if abstraction not in abstract_symbols:
                                abstract_symbols.add(abstraction)
                                self.state_2_one_hot_map[abstraction] = counter
                                counter += 1
                    else:
                        self.state_2_one_hot_map[(x, y)] = counter
                        counter += 1

        if self.rules:
            for state in list(self.state_2_one_hot_map.keys()):
                for act in self.actions_dict.keys():
                    slip_state = f'{state}_slip_{act}'
                    self.state_2_one_hot_map[slip_state] = counter
                    counter += 1

        if self.indicate_wall:
            for output in list(self.state_2_one_hot_map.keys()):
                self.state_2_one_hot_map[f'{output}_wall'] = counter
                counter += 1

        self.one_hot_2_state_map = {v: k for k, v in self.state_2_one_hot_map.items()}
        return counter

    def step(self, action):
        assert action in self.actions

        self.step_counter += 1

        new_location = self._get_new_location(action)
        if self.world[new_location[0]][new_location[1]] == '#':
            observation = self.get_observation()
            if self.indicate_wall:
                observation = f'{observation}_wall'
            done = True if self.step_counter >= self.max_ep_len else False
            return self.encode(observation), self.step_penalty, done, {}

        # If you open the door, perform that step once more and enter new room
        if self.world[new_location[0]][new_location[1]] == 'D':
            self.player_location = new_location
            new_location = self._get_new_location(action)

        # Update player location
        self.player_location = new_location

        # Account for behavioural toggle (disable/enable stochastic behaviour)
        if self.player_location in self.behavioral_toggles:
            self.use_stochastic_tiles = not self.use_stochastic_tiles

        # Reward is reached if goal is reached. This terminates the episode.
        reward = 0
        if self.player_location in self.reward_tiles.keys():
            if self.one_time_rewards and self.player_location not in self.collected_rewards:
                reward = self.reward_tiles[self.player_location]
            elif not self.one_time_rewards:
                reward = self.reward_tiles[self.player_location]
            self.collected_rewards.add(self.player_location)

        done = False

        if self.player_location in self.goal_locations:
            reward = self.goal_reward
            done = True
        if self.player_location in self.terminal_locations:
            reward = self.goal_reward * -1
            done = True

        if self.step_counter >= self.max_ep_len:
            done = True

        if self.step_penalty != 0 and reward == 0:
            reward = self.step_penalty

        observation = self.get_observation()

        return self.encode(observation), reward, done, {}

    def get_observation(self):
        if self.is_partially_obs:
            if self.indicate_slip and self.slip_action is not None:
                observation = f'{self.get_abstraction()}_slip_{self.slip_action}'
            else:
                observation = self.get_abstraction()
        else:
            observation = self.player_location
        return observation

    def _get_new_location(self, action):
        old_action = action
        self.slip_action = None

        if self.player_location in self.stochastic_tile.keys() and self.use_stochastic_tiles:
            action = self.rules[self.stochastic_tile[self.player_location]].get_action(action)
        if old_action != action:
            self.slip_action = self.action_space_to_act_map[action]
        return self.move(action)

    def move(self, action):
        if action == 0:  # up
            return self.player_location[0] - 1, self.player_location[1]
        if action == 1:  # down
            return self.player_location[0] + 1, self.player_location[1]
        if action == 2:  # left
            return self.player_location[0], self.player_location[1] - 1
        if action == 3:  # right
            return self.player_location[0], self.player_location[1] + 1

    def encode(self, state):
        return self.state_2_one_hot_map[state]

    def decode(self, one_hot_enc):
        return self.one_hot_2_state_map[one_hot_enc]

    def get_abstraction(self):
        abstract_tile = self.abstract_world[self.player_location[0]][self.player_location[1]]
        if abstract_tile != ' ':
            return self.abstract_symbol_name_map[abstract_tile]
        else:
            return self.player_location

    def reset(self):
        self.step_counter = 0
        self.slip_action = None
        self.use_stochastic_tiles = True
        self.player_location = self.initial_location[0], self.initial_location[1]
        self.collected_rewards.clear()
        return self.encode(self.get_observation())

    def render(self, mode='human'):
        world_copy = deepcopy(self.world)
        world_copy[self.player_location[0]][self.player_location[1]] = 'E'
        for l in world_copy:
            print("".join(l))

    def play(self):
        self.reset()
        user_input_map = {'w': 0, 's': 1, 'a': 2, 'd': 3}
        print('Agent is controlled with w,a,s,d; for up,left,down,right actions.')
        while True:
            self.render()
            action = input('Action: ', )
            output, reward, done, info = self.step(user_input_map[action])
            print(f'Output: {self.decode(output), reward, done, info}')
