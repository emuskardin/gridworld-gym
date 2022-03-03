# POGE - Partially Observable Gridworld Environment 

**POGE** is an gridworld environment generator for the [gym framework](https://gym.openai.com/). It puts special focus 
on creation of deterministic or stochastic gridworlds with an option to toggle partial observability. All created
environments can be [scaled to any size](world_scaler.py).
This environments can be used for testing and development of planning and classical RL algorithms.

Partial observability stems from abstraction over the state representation.
In future iteration user will be able to define non-Markovian goals.

**Working example** can be found in the `q_learning.py` file, where one can see how classic Q-learning often fails 
to learn good policy in the presence of the abstraction (of the state representation).

## World Generation 

Each gridworld is defined in its own .txt file. 

The file is divided into sections:
- Layout
- Abstraction
- Behaviour
- Rewards

Each section starts with the `===<section name>===`.

More details about each aspect of the world generation and enviroment initialization are presented after the example.

Reserved characters (used in Layout section):
- `#` - wall 
- `D` - door in the wall (player just passes through it)
- `E` - starting player location
- `G` - goal location(s)
- `T` - terminal state. If reached, negative goal reward will be given and episode will be terminated
- `@` - behavioural toggle. If this state is reached, all stochastic behaviour will become deterministic and vice versa.
## Example World
```
===Layout===

// E is ego, that is starting position and player indication when rendering
// G is the goal. Once it is reached, the episode terminates. There can be multiple goals.

#########
#   #   #
#   D  E#
#   #   #
##D######
#   # G #
#   D   #
#   #   #
#########

===Abstraction===

// If no abstraction is defined for a tile (tile is empty), x and y coordinates will be returned.

#########
#111#222#
#111D222#
#111#222#
##D######
#333#444#
#333D444#
#333#444#
#########

// Optional mapping of integers/characters to their description/name

2:corridor
3:living_room
4:toilet

===Behaviour===

#########
#1  # 3 #
# 2 D   #
#1  # 2 #
##D######
# 1  #2 #
#   D   #
# 2 #3  #
#########

// Rules
// <rule_id>-<action>-[(<new_action>:<probability>)+]

1-up-[up:0.75, right:0.25]
1-down-[down:0.75, right:0.25]
2-left-[left:0.8, up:0.1, down:0.1]
3-right-[right:0.7, down:0.2, left:0.1]

===Rewards===

#########
#   #   #
# a D   #
#*  #   #
##D######
#   #   #
# b D   #
#*  # *c#
#########

// Mapping of reward tile to reward value. Only positive and negative integers are possible.

a:1
b:2
c:3
*:-5

```

### Layout
*Layout* section is a necessity for minimal working examples.
All other sections are optional.
All sections should contain the same border structure.
In the *Layout* section, we define the layout of the environment and the starting, goal, and terminal positions.


*Rewards* define the reward structure, that is whether a positive or a negative reward is received once the player reaches 
a certain tile.

### Abstraction

*Abstraction* defines abstract outputs player will receive once he is on a tile. If no abstract output 
is given (for a tile or the whole layout), observations consist of x and y coordinates.
If the `is_partially_obs` parameter of the constructor is set to True (True by default), 
instead of observing x and y coordinates, a value found on the tile will be returned. 

### Behaviour

*Behaviour* defines the transition probabilities between tiles. 
If the section is not defined, the environment will be deterministic.

To add stochastic behavior to the environment we create "rules". 
Each tile of the environment can be assigned to a single (or no) rule.
If no rule is assigned to a tile the behavior of a tile will be deterministic.

In the rules file, first, we lay out the map found in the layout file and assign `rule_id` (integer or char) to tiles that we want to behave in a certain way.
Once the layout has been defined, we need to declare rules for each `rule_id`.
They are of the form

```
<rule_id>-<action>-[(<new_action>:<probability>)+]
```
where the action is the action we are trying to execute, and the new action is the action that can occur with declared probability.
If no action is specified for a rule, it will remain deterministic.

### Rewards

Intermediate rewards can be declared by assigning a symbol to a tile. Then the reward of the tile will equal the integer value mapping to the symbol.
Mapping of symbols to reward can be seen in the example. It follows the `<symbol>:<reward>` syntax.

If the `one_time_rewards` param is set to True (by default it is), then the reward for each tile will be received once per episode. 
Else it will be received every time player is on the reward tile.

## Initialize gridworld environment

```python
import gym
import gym_partially_observable_grid

# Make environment deterministic even if it is stochastic
force_determinism = False
# Add slip to the observation set (action failed). Only necessary if is_partially_obs is set to True AND you want
# the underlying system to behave like deterministic MDP. If is_partially_obs is set to False, this value is ignored.
indicate_slip = True
# Use abstraction/partial observability. If set to False, (x,y) coordinates will be used as outputs
is_partially_obs = False
# If one_time_rewards is set to True, reward in single location will be obtained only once per episode.
# Otherwise, reward will be given every time
one_time_rewards = True
# Step penalty defines a negative reward that is received every step if the reward is not obtained

env = gym.make(id='poge-v1', 
               world_file_path='worlds/world0.txt',
               force_determinism=force_determinism,
               indicate_slip=indicate_slip,
               is_partially_obs=is_partially_obs,
               one_time_rewards=one_time_rewards,
               step_penalty=-0.1)
```
