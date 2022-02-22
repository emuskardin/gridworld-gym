from collections import defaultdict
from random import choices


class StochasticTile:
    def __init__(self, rule_id):
        self.rule_id = rule_id
        self.behaviour = dict()

    def add_stochastic_action(self, action, new_action_probabilities):
        self.behaviour[action] = new_action_probabilities
        assert round(sum([p[1] for p in new_action_probabilities]), 5) == 1.0

    def get_action(self, action):
        if action not in self.behaviour.keys():
            return action
        actions = [a[0] for a in self.behaviour[action]]
        prob_dist = [a[1] for a in self.behaviour[action]]

        new_action = choices(actions, prob_dist, k=1)[0]

        return new_action

    def get_all_actions(self):
        return list({action_prob_pair[0] for rule in self.behaviour.values() for action_prob_pair in rule})


class PartiallyObsGridworldParser:
    def __init__(self, path_to_file):
        self.content = defaultdict(list)

        # State space
        self.state_space = 0
        # Representation of concrete ((x,y) coordinates) world and abstract world
        self.world, self.abstract_world = None, None
        # Map of stochastic tiles, where each tile is identified by rule_id
        self.rules = dict()
        # Map of locations to rule_ids, that is, tile has stochastic behaviour
        self.stochastic_tile = dict()
        # Map of locations that return a reward
        self.reward_tiles = dict()
        # Map of symbols to rewards
        self.symbol_reward_map = dict()
        # Map of abstract symbols to their names (if any)
        self.abstract_symbol_name_map = dict()

        # Variables
        self.initial_location = None
        self.player_location = None
        self.goal_location = set()
        self.terminal_locations = set()
        self.behavioral_toggles = set()

        self._parse_file(path_to_file)
        self._parse_world_and_abstract_world()
        self._parse_abstraction_mappings()
        self._parse_layout_variables()
        self._parse_rewards()
        self._parse_rules()

    def _parse_file(self, path_to_file):
        file = open(path_to_file, 'r')
        current_section = None
        sections = ['Layout', 'Abstraction', 'Behaviour', 'Rewards']
        for line in file.readlines():
            line = line.strip()
            if not line or line.startswith('//'):
                continue
            is_header = False
            for section_name in sections:
                if section_name in line:
                    current_section = section_name
                    is_header = True
            if is_header:
                continue

            self.content[current_section].append(line)

    def _parse_world_and_abstract_world(self):
        self.world = [list(l.strip()) for l in self.content['Layout']]
        self.abstract_world = [list(l.strip()) for l in self.content['Abstraction'][:len(self.world)]]
        if not self.content['Abstraction']:
            self.abstract_world = None
        else:
            assert len(self.world) == len(self.abstract_world)

    def _parse_abstraction_mappings(self):
        abstract_tiles = set()
        for i, line in enumerate(self.content['Abstraction']):
            if i <= len(self.world) - 1:
                for abstract_tile in list(line):
                    if abstract_tile not in {'#', 'D', 'G', 'E', ' '}:
                        abstract_tiles.add(abstract_tile)

            # add custom names
            else:
                id_name_pair = line.split(':')
                id, name = id_name_pair[0], id_name_pair[1]
                self.abstract_symbol_name_map[id] = name

        # if some abstract tile does not have defined name, it will just be itself :)
        for at in list(abstract_tiles):
            if at not in self.abstract_symbol_name_map.keys():
                self.abstract_symbol_name_map[at] = at

    def _parse_layout_variables(self):
        for x, line in enumerate(self.content['Layout']):
            for y, tile in enumerate(line):
                if tile == 'E':
                    self.player_location = (x, y)
                    self.initial_location = (x, y)
                    self.world[self.player_location[0]][self.player_location[1]] = ' '
                if tile == 'G':
                    self.goal_location.add((x, y))
                if tile == 'T':
                    self.terminal_locations.add((x, y))
                if tile == '@':
                    self.behavioral_toggles.add((x, y))
                if tile != '#' and tile != 'D':
                    self.state_space += 1

        assert self.player_location and self.goal_location

    def _parse_rewards(self):
        for x, line in enumerate(self.content['Rewards']):
            if x <= len(self.world) - 1:
                for y, tile in enumerate(line):
                    if tile not in {'#', 'D', 'G', ' '}:
                        self.reward_tiles[(x, y)] = tile
            else:
                symbol_value_pair = line.split(':')
                symbol, value = symbol_value_pair[0], symbol_value_pair[1]
                self.symbol_reward_map[symbol] = int(value)

        for k, v in self.reward_tiles.items():
            self.reward_tiles[k] = self.symbol_reward_map[v]

    def _parse_rules(self):

        # Extract the rules layout
        rule_world = []
        for index, line in enumerate(self.content['Behaviour']):
            # First part of the rules section corresponds to a map
            if index <= len(self.world) - 1:
                rule_world.append(line)
            else:
                self._parse_and_process_rule(line)

        for x, line in enumerate(rule_world):
            for y, tile in enumerate(line):
                if tile in self.rules.keys():
                    tile_xy = (x, y)
                    self.stochastic_tile[tile_xy] = tile

    def _parse_and_process_rule(self, rule):
        actions_dict = {'up': 0, 'down': 1, 'left': 2, 'right': 3}

        rule = ''.join(rule)
        rule = rule.replace(" ", '')
        rule_parts = rule.split('-')
        rule_id = rule_parts[0]
        rule_action = actions_dict[rule_parts[1]]
        rule_mappings = rule_parts[2]
        rule_mappings = rule_mappings.lstrip('[').rstrip(']')
        if rule_id not in self.rules.keys():
            self.rules[rule_id] = StochasticTile(rule_id)

        action_prob_pairs = []
        for action_prob in rule_mappings.split(','):
            ap = action_prob.split(':')
            action = actions_dict[ap[0]]
            prob = float(ap[1])
            action_prob_pairs.append((action, prob))

        self.rules[rule_id].add_stochastic_action(rule_action, action_prob_pairs)
