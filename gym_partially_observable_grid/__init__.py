from gym.envs.registration import register

register(
     id='poge-v1',
     entry_point='gym_partially_observable_grid.envs:PartiallyObservableWorld',
 )