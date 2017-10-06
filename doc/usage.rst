Usage
=====

model
-----

`oedes` model contains the discrete system equation to be solved. It is built using `mesh`, which defines spatial discretization.

params
------

`params` is a dictionary of parameter values. Keys which are specific to part of the model start with a prefix, which refers to species name and also the layer. All numeric values are given in SI base units, except for small energies which are specified in eV.

Summary of most common parameters is given below (`*` denotes prefix):

- `T` [K]: ambient temperature in fixed temperature simulation
- `*.mu` [m^2/(Vs)]: charge carrier mobility, `*` is charge carrier
- `*.level` [V]: HOMO/LUMO level, -E/e, `*` is charge carrier
- `*.voltage` [V]: applied voltage, `*` is electrode
- `*.N0` [1/(m^3)]: total density of states, `*` is electronic charge carrier
- `*.workfunction` [V]: workfunction divided by elementary charge W/e, `*` is electrode 
- `npi` [1/(m^6)]: intrinsic product of electron and hole concentration

Currently, no default values are assumed and calculation fails if any parameter is not specified.

solver
------

Solver calculates the solution vector for `model` and concrete values of `params`.

outputs
-------

Summary of the most important outputs is given below (`*` denotes `prefix`):

- `*.c` [1/m^3]: concentration of species `*`, for each cell
- `*.ct` [1/(m^3 s)]: time derivative of concentration of species `*`, for each cell
- `*.j` [1/(m^2 s)]: flux of species `*`, for each face
- `*.jdrift` [1/(m^2 s)]: drift (advection) part of flux of species `*`, for each face
- `*.jdiff` [1/(m^2 s)]: diffusion (advection) part of flux of species `*`, for each face
- `R` [1/(m^3 s)]: recombination density, for each cell
- `J` [A/m^2]: total electric current density
- `E` [V/m]: electric field, for each face
- `*.potential` [V]: electrostatic potential, for each cell

context
-------

context object connects the model, parameters, solution vector, and outputs. It also provides support for simple plotting.
