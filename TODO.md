# Remaining Implementations (by file)

## `src/individual.py`

**Implemented**

- ✓ `Individual.__init__()`
- ✓ `Individual.create_random_nodes()`
- ✓ `Individual.create_output_layer()`
- ✓ `Individual.realize_model()`
- ✓ `Individual.evaluate_fitness()`
- ✓ `Individual.clone()`
- ✓ `Individual.mutate(..., add_layers=False, swap_layers=False, delete_layers=False)`
- ✓ `Individual.mutate_parameters()`

**TODO**

The options:

- ✗ `Individual.mutate(..., add_layers=True, swap_layers=False, delete_layers=False)`
- ✗ `Individual.mutate(..., add_layers=False, swap_layers=True, delete_layers=False)`
- ✗ `Individual.mutate(..., add_layers=False, swap_layers=False, delete_layers=True)`

which must be implemented by:

- ✗ `Individual.mutate_new_layer()`
- ✗ `Individual.mutate_delete_layer()`
- ✗ `Individual.mutate_swap_layer()`

## `src/population.py`

**Implemented**

- ✓ `Population.__init__()`
- ✓ `Population.evaluate_fitnesses()`

**TODO**

- ✗ `Population.clone()`
- ✗ `Population.mutate()`
- ✗ `Population.select_best()`
- ✗ `Population.get_crossover_pairs()`
  - ✗ `Population.get_crossover_pairs(method="random")`
  - ✗ `Population.get_crossover_pairs(method="best")`
  - ✗ `Population.get_crossover_pairs(method="weighted-random")`
- ✗ `Population.crossover()`

need to be implemented, which means implementing:

- ✗ `Individual.mutate_new_layer()`
- ✗ `Individual.mutate_delete_layer()`
- ✗ `Individual.mutate_swap_layer()`

## `src/crossover.py`

This isn't needed right now to get things working, since you can call construt a `Generation` that
does not use crossover during evolution, but we do want this implemented.

**TODO**

- ✗ `cross_individuals()`


## `src/generation.py`

**Implemented**

- ✓ `HallOfFame.__init__()`

- ✓ `Generation.__init__()`
- ✓ `Generation.next()`
- ✓ `Generation.evaluate_fitnesses()`
- ✓ `Generation.get_survivors()`
- ✓ `Generation.set_offspring()`

**Implemented when Dependencies are Implemented**
- ✓ `Generation.save_progress()`
  - Dependencies: `Individual.save`, `Population.save`, `HallOfFame.update`
- ✓ `Generation.mutate_survivors()`
  - Dependencies: all `Individual.mutate_...` options and functions
- ✓ `Generation.cross()`
  - Dependencies: `Population.crossover()`, and `crossover()` in `src/crossover.py`

**TODO**

- ✗ `HallOfFame.save()`
- ✗ `HallOfFame.update()`
  - **Note**: this will be easy when `Individual.save` and `Population.save` are implemented

- ✗ `Generation.update_hall_of_fame()`
- ✗ `Generation.filter_by_fitness()`


# Tests that Must be Passed

There are numerous tests that must always pass for all code. The most basic of these tests can be
run with

```sh
pytest -m spec
```
or

```sh
python -m pytest -m spec
```
where `python` is a python interpreter installed in the virtual environment for this project
(Note: `pytest -m <mark>` runs all tests that have been *m*arked with `<mark>`).


If you get errors run running `pytest -m spec`, these errors must be resolved before preceding with
current code. Most of the time, this will mean there is an error in one of your implementations, but
sometimes it will be because there is an error in one of the tests. In the latter case you should
contact me.


