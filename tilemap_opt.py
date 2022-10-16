import time
from enum import Enum
import numpy as np
from mlrose_hiive.opt_probs.discrete_opt import DiscreteOpt
from mlrose_hiive.fitness.custom_fitness import CustomFitness

class Ordinal(Enum):
    NORTH = 0,
    EAST = 1,
    SOUTH = 2,
    WEST = 3

    def from_char(char):
        if char[0] == 'n' or char[0] == 'N':
            return Ordinal.NORTH
        elif char[0] == 'e' or char[0] == 'E':
            return Ordinal.EAST
        elif char[0] == 's' or char[0] == 'S':
            return Ordinal.SOUTH
        elif char[0] == 'w' or char[0] == 'W':
            return Ordinal.WEST
        else:
            raise Exception(f'Invalid ordinal character: {char}')

    # (y, x)
    def offset(self):
        if self == Ordinal.NORTH:
            return (-1, 0)
        elif self == Ordinal.EAST:
            return (0, 1)
        elif self == Ordinal.SOUTH:
            return (1, 0)
        elif self == Ordinal.WEST:
            return (0, -1)
        else:
            raise NotImplementedError

    def opposite(self):
        if self == Ordinal.NORTH:
            return Ordinal.SOUTH
        elif self == Ordinal.EAST:
            return Ordinal.WEST
        elif self == Ordinal.SOUTH:
            return Ordinal.NORTH
        elif self == Ordinal.WEST:
            return Ordinal.EAST
        else:
            raise NotImplementedError

class TilemapConstraints:
    def __init__(self, tiles, constraints):
        self.tiles = tiles
        self.constraints = constraints
        self.tile_indices = {}

        for i, tile in enumerate(self.tiles):
            assert isinstance(tile, str)
            self.tile_indices[tile] = i
            if tile not in self.constraints:
                self.constraints[tile] = {}
            for ordinal in Ordinal:
                assert isinstance(self.constraints[tile], dict)
                if ordinal not in self.constraints[tile]:
                    self.constraints[tile][ordinal] = set()

    # Rules are in the form of (tile, ordinal, tile)
    def from_rules(rules):
        unique_tiles = set()
        tiles = []
        constraints = {}
        for rule in rules:
            assert len(rule) == 3
            tile1, ordinal, tile2 = rule

            # Add tiles
            for tile in [tile1, tile2]:
                if tile not in unique_tiles:
                    tiles.append(tile)
                    unique_tiles.add(tile)

            constraints[tile1] = constraints.get(tile1, {})
            constraints[tile2] = constraints.get(tile2, {})

            ordinal = [Ordinal.from_char(c) for c in ordinal]

            for o in ordinal:
                constraints[tile1][o] = constraints[tile1].get(o, set())
                constraints[tile2][o.opposite()] = constraints[tile2].get(o.opposite(), set())
                constraints[tile1][o].add(tile2)
                constraints[tile2][o.opposite()].add(tile1)

        return TilemapConstraints(tiles, constraints)

    def stringify(self, state):
        return [self.tiles[s] for s in state]

class TilemapRasterizer:
    def __init__(self, problem, tile_images):
        assert isinstance(tile_images, dict)
        assert isinstance(problem, TilemapGeneration)

        tile_size = None
        for tile in tile_images:
            tile_size = tile_size or tile_images[tile].shape
            assert tile_images[tile].shape == tile_size

        self.tile_images = tile_images
        self.tile_size = tile_size
        self.problem = problem

    def rasterize(self, state):
        assert len(state) == self.problem.size[0] * self.problem.size[1]
        img = np.zeros((self.problem.size[0] * self.tile_size[0], self.problem.size[1] * self.tile_size[1], self.tile_size[2]))
        
        for i, tile_index in enumerate(state):
            y = i // self.problem.size[1]
            x = i % self.problem.size[1]

            y1 = y * self.tile_size[0]
            y2 = y1 + self.tile_size[0]
            x1 = x * self.tile_size[1]
            x2 = x1 + self.tile_size[1]
            img[y1:y2, x1:x2, :] = self.tile_images[self.problem.constraints.tiles[tile_index]]
        return img

class TilemapFitness:
    def __init__(self, size, constraints):
        self.prob_type = 'discrete'
        self.size = size
        self.constraints = constraints

        self.one_hots = {} # Contains map of ordinal => n x n array
        for ordinal in Ordinal:
            one_hots = []
            for tile in constraints.tiles:
                one_hot = np.zeros((len(constraints.tiles),))
                for allowed_tile in constraints.constraints[tile][ordinal]:
                    one_hot[constraints.tile_indices[allowed_tile]] = 1
                one_hots.append(one_hot)
            self.one_hots[ordinal] = np.array(one_hots)

    def get_prob_type(self):
        return self.prob_type

    def one_hot(self, values, maxval=None):
        maxval = np.max(values) + 1 if maxval is None else maxval
        return np.eye(maxval)[values]

    def ordinal_slices(self, ordinal):
        oy, ox = ordinal.offset()
        sy = max(-oy, 0)
        ey = min(self.size[0] - oy, self.size[0])

        sx = max(-ox, 0)
        ex = min(self.size[1] - ox, self.size[1])

        return slice(sy, ey), slice(sx, ex)

    def ordinal_allowance(self, state, ordinal):
        oy, ox = ordinal.offset()
        
        allowed = np.take(self.one_hots[ordinal], state, axis=0).reshape(self.size + (-1,))
        result = np.ones(allowed.shape)

        yslice, xslice = self.ordinal_slices(ordinal)
        allowed = allowed[yslice, xslice, :]

        yslice, xslice = self.ordinal_slices(ordinal.opposite())
        result[yslice, xslice, :] = allowed

        return result

    def evaluate(self, state):
        st = time.time()
        allowed_by_ordinal = []
        for ordinal in Ordinal:
            allowed = self.ordinal_allowance(state, ordinal)
            allowed_by_ordinal.append(allowed)

        one_hot_state = self.one_hot(state, len(self.constraints.tiles)).reshape(self.size + (-1,))
        allowed_by_ordinal.append(one_hot_state)
        allowed_by_ordinal = np.logical_and.reduce(allowed_by_ordinal, axis=0)
        allowed_by_ordinal = np.max(allowed_by_ordinal, axis=-1)
        f1 = allowed_by_ordinal.mean()
        et = time.time()
        t1 = et - st
        st = time.time()
        f2 = tilemap_fitness(state, self.size, self.constraints)
        et = time.time()
        t2 = et - st
        print(f"t1={t1}, t2={t2} | ratio={t1 / t2}")
        assert f1 == f2, f"f1={f1} != f2={f2}"
        return f1

    def evaluate_many(self, states):
        raise NotImplementedError


# fitness: [0, 1.0]
def tilemap_fitnessV1(state, size, constraints):
    valid_tile_count = 0
    for y, x in np.ndindex(size):
        i =  y * size[1] + x
        ti = int(state[i])
        tile = constraints.tiles[ti]
        is_valid = True
        for ordinal in Ordinal:
            oy, ox = ordinal.offset()
            y2, x2 = y + oy, x + ox
            if y2 < 0 or x2 < 0 or y2 >= size[0] or x2 >= size[1]:
                continue # Skip out of bounds
            j = y2 * size[1] + x2
            other_tile = constraints.tiles[int(state[j])]
            if other_tile not in constraints.constraints[tile][ordinal]:
                is_valid = False
                break
            # probably unnecessary check
            if tile not in constraints.constraints[other_tile][ordinal.opposite()]:
                is_valid = False
                break

        if is_valid:
            valid_tile_count += 1
    return valid_tile_count / (size[0] * size[1])

def tilemap_fitness(state, size, constraints):
    valid_tile_count = 0
    for y, x in np.ndindex(size):
        i =  y * size[1] + x
        ti = int(state[i])
        tile = constraints.tiles[ti]
        is_valid = True
        for ordinal in Ordinal:
            oy, ox = ordinal.offset()
            y2, x2 = y + oy, x + ox
            if y2 < 0 or x2 < 0 or y2 >= size[0] or x2 >= size[1]:
                continue # Skip out of bounds
            j = y2 * size[1] + x2
            other_tile = constraints.tiles[int(state[j])]
            if other_tile not in constraints.constraints[tile][ordinal]:
                is_valid = False
                break
            # probably unnecessary check
            if tile not in constraints.constraints[other_tile][ordinal.opposite()]:
                is_valid = False
                break

        if is_valid:
            valid_tile_count += 1
    score = valid_tile_count / (size[0] * size[1])
    if score < 1.0:
        return score
    else:
        return score + (np.unique(state).flatten().shape[0] / (size[0] * size[1]))

class TilemapGeneration(DiscreteOpt):
    def __init__(self, size, constraints, **kwargs):
        assert len(size) == 2
        assert isinstance(size, tuple)
        assert isinstance(constraints, TilemapConstraints)
        self.size = size
        self.constraints = constraints
        super().__init__(
            length=size[0]*size[1],
            fitness_fn=CustomFitness(tilemap_fitness, size=self.size, constraints=self.constraints),
            #fitness_fn=TilemapFitness(self.size, self.constraints),
            maximize=True,
            max_val=len(constraints.tiles),
            **kwargs)
