# limepy
Python tools to run and analyse LIME models.

## Input
Requires a header file readable by C containing (at least) the arrays named
`c1arr`, `c2arr`, `dens`, `temp` and `abund` where the first two are either the
cylindrical coordinates (r, z) or polar coordinates (r, theta). A third
dimension can be added with `c3arr`, which is the azimuthal angle. With this,
`limepy` will write the necessary `model.c` files and execute them.


## Running
A standard call will be something like:

```python
from limepy.model.runlime import run_model

run_model(header='header.h',
          name='filenames',
          molecule='13co',
          trans=[1, 2, 3],
          incl=[0.4],
          )
```

Most of the variables from LIME are able to be included. If lists of
transitions, inclinations or position angles, all permutations will be made.


In order to increase the signal to noise of the data, one can run several
models and then average over them using the `nmodels` keyword.
