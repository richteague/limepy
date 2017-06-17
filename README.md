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
