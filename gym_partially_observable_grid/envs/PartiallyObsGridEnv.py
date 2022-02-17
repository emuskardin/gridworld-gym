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
                 max_ep_len=100,
                 goal_reward=10,
                 one_time_rewards=True):

        # Available actions
        self.actions_dict = {'up': 0, 'down': 1, 'left': 2, 'right': 3}
        self.actions = [0, 1, 2, 3]

        parser = PartiallyObsGridworldParser(world_file_path)

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
        self.last_action_slip = False

        # Indicate whether observations will be abstracted or will they be x-y coordinates
        self.is_partially_obs = is_partially_obs

        # If abstraction is not defined, environment cannot be partially observable
        if self.abstract_world is None:
            self.is_partially_obs = False

        # Variables
        self.initial_location = parser.initial_location
        self.player_location = parser.player_location
        self.goal_location = parser.goal_location

        # Reward reached when reaching goal
        self.goal_reward = goal_reward

        # Episode lenght
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
                slip_state = f'{state}_slip'
                self.state_2_one_hot_map[slip_state] = counter
                counter += 1

        self.one_hot_2_state_map = {v: k for k, v in self.state_2_one_hot_map.items()}
        return counter

    def step(self, action):
        assert action in self.actions

        self.step_counter += 1
        new_location = self._get_new_location(action)
        if self.world[new_location[0]][new_location[1]] == '#':
            observation = self.get_observation()
            done = True if self.step_counter >= self.max_ep_len else False
            return self.encode(observation), 0, done, {}

        # If you open the door, perform that step once more and enter new room
        if self.world[new_location[0]][new_location[1]] == 'D':
            self.player_location = new_location
            new_location = self._get_new_location(action)

        # Update player location
        self.player_location = new_location

        # Reward is reached if goal is reached. This terminates the episode.
        reward = 0
        if self.player_location in self.reward_tiles.keys():
            if self.one_time_rewards and self.player_location not in self.collected_rewards:
                reward = self.reward_tiles[self.player_location]
            elif not self.one_time_rewards:
                reward = self.reward_tiles[self.player_location]
            self.collected_rewards.add(self.player_location)

        if self.player_location in self.goal_location:
            reward = self.goal_reward

        done = 1 if self.player_location in self.goal_location or self.step_counter >= self.max_ep_len else 0
        observation = self.get_observation()

        return self.encode(observation), reward, done, {}

    def get_observation(self):
        if self.is_partially_obs:
            if self.indicate_slip and self.last_action_slip:
                observation = self.get_abstraction() + '_slip'
            else:
                observation = self.get_abstraction()
        else:
            observation = self.player_location
        return observation

    def _get_new_location(self, action):
        old_action = action
        self.last_action_slip = False
        if self.player_location in self.stochastic_tile.keys():
            action = self.rules[self.stochastic_tile[self.player_location]].get_action(action)
        if old_action != action:
            self.last_action_slip = True
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
        while True:
            action = int(input('Action (up:0, down:1, left:2, right:3}): '))
            o = self.step(action)
            self.render()
            print(o)
