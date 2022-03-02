from collections import defaultdict

from gym_partially_observable_grid.envs import PartiallyObservableWorld


def parse_file(path_to_file):
    content = defaultdict(list)
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

        content[current_section].append(line)

    layout = [list(line) for line in content['Layout']]
    abstraction_layout = [list(line) for line in content['Abstraction']][:len(layout)]

    abstraction_mappings = ''
    for am in content['Abstraction'][len(layout):]:
        abstraction_mappings += f'{am}\n'

    behaviour_layout = [list(line) for line in content['Behaviour']][:len(layout)]
    behaviour_mapping = ''
    for rm in content['Behaviour'][len(layout):]:
        behaviour_mapping += f'{rm}\n'

    rewards_layout = [list(line) for line in content['Rewards']][:len(layout)]
    rewards_mappings = ''
    for rm in content['Rewards'][len(layout):]:
        rewards_mappings += f'{rm}\n'

    for l in [layout, abstraction_layout, behaviour_layout, rewards_layout]:
        if not l:
            continue
        l.pop(0)
        l.pop()
        for line in l:
            line.pop(0)
            line.pop()

    return layout, abstraction_layout, abstraction_mappings, behaviour_layout, behaviour_mapping, rewards_layout, rewards_mappings


def create_world(file_name, parsed_values, repeat_x, repeat_y):
    layout, abstraction_layout, abstraction_mappings, rules_layout, rules_mappings, rewards_layout, rewards_mappings = parsed_values

    layout_world = []
    abstract_world = []
    behaviour_world = []
    rewards_world = []

    for world, word_list in [(layout_world, layout), (abstract_world, abstraction_layout),
                             (behaviour_world, rules_layout), (rewards_world, rewards_layout)]:
        if not word_list:
            continue

        curr_row = 0
        last_row = repeat_y * len(layout) - 1
        for r_y in range(repeat_y * len(layout)):

            line = []
            for x in range(repeat_x):
                line.extend(word_list[curr_row % len(layout)])

            world.append(line)

            curr_row += 1

        for line in world:
            line.insert(0, '#')
            line.append('#')

        e_set, last_g_location = False, None
        for x, line in enumerate(layout_world):
            for y, tile in enumerate(line):
                if tile == 'G':
                    last_g_location = (x, y)

        for x, line in enumerate(layout_world):
            for y, tile in enumerate(line):
                if tile == 'E':
                    if not e_set:
                        e_set = True
                        continue
                    else:
                        layout_world[x][y] = ' '
                if tile == 'G' and (x, y) != last_g_location:
                    layout_world[x][y] = ' '

        top_bottom_line = ['#' for _ in range(len(world[0]))]
        world.insert(0, top_bottom_line)
        world.append(top_bottom_line)

    with open(file_name, 'w') as file:

        file.write('===Layout===\n\n')
        for l in layout_world:
            file.write(''.join(l)+'\n')

        file.write('\n')

        if abstract_world:
            file.write('===Abstraction===\n\n')
            for l in abstract_world:
                file.write(''.join(l)+'\n')
            file.write('\n' + abstraction_mappings + '\n')

        if behaviour_world:
            file.write('===Behaviour===\n\n')
            for l in behaviour_world:
                file.write(''.join(l)+'\n')
            file.write('\n' + rules_mappings + '\n')

        if rewards_world:
            file.write('===Rewards===\n\n')
            for l in rewards_world:
                file.write(''.join(l)+'\n')
            file.write('\n' + rewards_mappings + '\n')


if __name__ == '__main__':
    p = parse_file('worlds/world0.txt')
    create_world('worlds/scaled_world0.txt', p, 3, 2)
