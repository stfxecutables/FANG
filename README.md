Automatically generate, select, and evolve deep learning architectures.

- [Code Design and Design Philosophy (For Students new to Python)](#code-design-and-design-philosophy-for-students-new-to-python)
  - [The Evolutionary Algorithm](#the-evolutionary-algorithm)
    - [Common Evolutionary Functions](#common-evolutionary-functions)
  - [Interfacing with Deep Learning Libraries](#interfacing-with-deep-learning-libraries)
    - [Interfacing with Layers](#interfacing-with-layers)
      - [The `Arguments` and `ArgDef` Class](#the-arguments-and-argdef-class)
    - [Interfacing with `Module`s and `initializer.py`](#interfacing-with-modules-and-initializerpy)
- [Current Implementation](#current-implementation)
  - [TODO](#todo)

# Code Design and Design Philosophy (For Students new to Python)

FANG really has only two components: the *evolutionary algorithm*, and *interfacing* with existing
deep learning libraries (PyTorch, Tensorflow) so that we can manipulate those libraries with code.

## The Evolutionary Algorithm

Implementing a general evolutionary algorithm requires implementing only a tiny handful of classes
and methods. Broadly, you need to implement a `Population` or `Generation` class, and an
`Individual` class.

The `Population` class will need methods like

  - `select() -> Population` - for choosing / identifying which individuals have survived and/or reproduced
  - `reproduce() -> Population` - for having the individuals reproduce (mutate + crossover)
  - `next() -> Population` - for returning / setting up a new population

or other similar general methods. Additionally, it may be worth thinking about making a distinction
between a `Population` and a `Generation`. E.g. a population might just be *any* grouping of
individuals, but a `Generation` might contain many populations (e.g. a `Generation` might have h
`Population`s like the *survivors*, *mating pool*, *children*, and *strongest so far*, for example)

 By contrast, the individual will need methods like:

  - `clone() -> Individual` - for copying the individual
  - `mutate(prob: float) -> Individual` - for mutating values at probability `prob`
  - `cross(individual2: Individual) -> Individual` for crossing with another individual

### Common Evolutionary Functions

If you think ahead about designing this, you can see we have a natural hierarchy. Generations
contain populations, populations contain individuals, individuals contain layers, and layers contain
arguments. To mutate a generation, we mutate each population, and to mutate each population, we
mutate each individual, and to mutate an individual, we mutate the layers, and mutating the layers
is just mutating the arguments (mostly).

It is rare that a programming problem ever has such a natural hierarchy, but since this does, an
object-oriented approach is going to work *very* well here and save you repeating a *tonne* of code.

The most important classes to create will be a `Layer` class which all other layers will subclass.
This is defined in `src/interface/pytorch/nodes/layer.py`.

However, based on above we know that various classes all need to e.g. `.mutate()`. Since many
classes need to mutate, the right thing to do is create a
[*mixin*](https://en.wikipedia.org/wiki/Mixin) for this. Mixins are really pretty simple, they just
either define which functions are needed, or implement standard implementations that work well in
most cases, and need to be over-written only in some cases.

The main mixin for FANG is called `Evolver` and is in `src/interface/pytorch/evolver.py`, and it
specifies only two functions `.mutate()` and `.clone()`. Almost everything in FANG needs to be an
`Evolver`, except perhaps Populations / Generations (but it makes sense to make these evolvers too).


## Interfacing with Deep Learning Libraries

Since an `Individual` in our case is a neural network, and the `Individual`s are composed of nodes
(layers), we are going to need to interface with deep-learning (DL) library network and layer
objects. For Tensorflow (Keras) and PyTorch, the general network class is the `Module`
(`tf.keras.Module` or `torch.nn.Module`), and layers are found in either `tf.keras.layers` for
Tensorflow, or `torch.nn` for PyTorch.

### Interfacing with Layers

In general, we don't really need to have a custom interface to the `Module` classes, because we can
just work with those directly using normal Python. However, there are *many* different layers, and
so we need a way to interact with all radically different layers in the exact same way. Thus, we
need an abstract interface for interacting with layers.

However to start, the only things we *really* need to do with layers are:

1. Give them random arguments
2. Mutate the arguments
3. Copy them
4. Initialize / create them
5. Join them into a network

The final three tasks are trivial, and so interfacing with layers largely just means working with
function arguments. Because Python allows `dict`s to be used as function arguments via the `**`
operator, this is thankfully extremely easy.

#### The `Arguments` and `ArgDef` Class

Once you realize we mostly only really need to interface with layer arguments, all you need to do is
read through the various layer documentations and see if there are any patterns to the arguments
that will let us abstract across them.

One thing that is clear for all programming languages is that arguments have only three components:
namely, a *name*, a *type*, and a *set of valid values* or *domain*. In PyTorch, you can see that
you can simplify things to only three or four types: all arguments can accept one of a single
`int`, a single `float`, or a single item from a small finite set. Additionally, some arguments can
accept a tuple of the previous options.

That means that you can fully *define* an argument with just three pieces of information. This is
what the `ArgDef` class in `src/interface/arguments.py` does.

Then, since Python also usually has sensible defaults for most parameters, you can define the *full
set of valid arguments* for a function with just a list of these `ArgDef` argument definitions.
Thus, the `Arguments` class in `src/interface/arguments.py` does this.

It is also worth noting that the `Arguments` class thus in a sense defines an implicit code, and a
way to save a network. Each ArgDef is easily JSON-able, and so an `Argument` instance is also easily
JSON-able. This makes saving abstract architectures very easy later, if we want to.


### Interfacing with `Module`s and `initializer.py`

Ultimately, we want to think abstractly in terms or networks, layers, individuals and populations,
and not have to worry about Tensorflow or PyTorch specifics. Thus, most of the time we want to be
working with *our* abstract classes.

However, we do ultimately have to translate *some* of our objects to Tensorflow or PyTorch objects.
This is where `src/interface/initializer.py` comes in. In this file, there are two mixins, `PyTorch`
and `Tensorflow`, both with only one method, `.create()`. This means the two classes conflict, and
an object can only be one or the other. This is intentional, because a Layer ultimately can only be
implemented in one or another framework for now. Likewise, a network can only be implemented in one
framework.

Thus the `.create()` method for the `PyTorch` should define how to get a PyTorch object, and the
`.create()` method for `Tensorflow` (not currently implemented) defines how to get the actual
Tensorflow object.


# Current Implementation

Right now, only the `Individual` is implemented, and no `Population` or `Generation` classes are
available. The `Individual` implementation is in the file `src/interface/individual.py`.

Multiple `Layer` interfaces, as well as the `Layer` base class are implemented for PyTorch in
`src/interface/pytorch/nodes`.

You will also notice that there is a file `src/interface/pytorch/optimizer.py`, which is *not* in
the `nodes` folder. You will have to think carefully about why the optimizer is quite separate and
not really a `Layer`, and be very careful about mutating this (maybe we don't want to mutate it as
much). Maybe it shoudn't even be an `Evolver` at all.

## TODO

We still need to implement:

- `Individual.mutate()` (about 90% done already; for mutations that just mutate valid parameter
  values)
- `Individual.mutate_structure()` which allows mutations that e.g. randomly delete and/or add and/or
  swap a layer
- `crossover(ind1: Individual, ind2: Individual) -> Individual` (challenging to ensure validity)
- a `Population` or `Generation` class with various methods like `select`, `save_best_individuals`,
  `next_generation` and etc
- model saving in a way that doesn't explode memory (easy, just need naming schemes based on time)
- many other layers. Currently only implemented are:
  - activations: ELU, Hardswish, LeakyReLU, PReLU, ReLU
  - convolutions: Conv2d, UpConv2d (Transposed Convolution)
  - dropout: Dropout, Dropout2d (drops the entire channel instead)
  - linear: Linear (aka Dense or Fully-Connected)
  - normalization: BatchNorm2d, InstanceNorm2d
  - pooling: MaxPool2d, AveragePool2d
- Various efficiency tasks and decisions re: mutation and generation of new individuals
  - e.g. fully mutating the optimizer params after selecting based on fitness is extremely unwise,
    since this makes the previous fitness invalid
  - whether or not to transfer over layer parameter values when mutating or crossing over (minimize
    required re-training length, save precious time)
  - properly handling training times dynamically for each model (learning rate scheduling,
    adjustment, and saving)
  - properly distributing activations among random choices (only 2 conv layers to choose from, but
    like 30 activations + normalizations yield useless networks of all activations and
    normalizations)
  - much much more



