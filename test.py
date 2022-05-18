from gym_partially_observable_grid.utils import PartiallyObsGridworldParser

parser = PartiallyObsGridworldParser('worlds/world1.txt')

model = parser.to_mdp()
model.visualize()