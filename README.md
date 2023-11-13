<img  src="assets/header_readme.gif"  />
<hr>
<p  align="center">
<b  style="font-size:30vw;">Montefiore AI - Fluid Simulator (maifs)</b>
</p>
<hr>

This a simple library to make fluid simulations and more ! With it, you will be able to:

- Generate datasets solving Burger (1D), Navier-Stokes (2D), ...
- Generate datasets easily on clusters (SLURM)

<hr>
<p  align="center">
<b  style="font-size:30vw;">Installation</b>
</p>
<hr>

First of all, you need to **clone** this repository:

```
git clone https://github.com/VikVador/Montefiore-AI---FluidSimulator.git
```

Then, you need to install the *maifs* library using this command at the root of the folder:

```
pip install -e .
```

From now on, you should be able to import *maifs*. If you want to install all the dependencies, it is possible to install a Conda environment as follows:

```
conda env create -f environment.yml
```

You  can activate the environment  with:

```
conda activate maifs
```

<hr>
<p  align="center">
<b  style="font-size:30vw;">Navier-Stokes 2D - Illustration </b>
</p>
<hr>

First of all, let us walk through an example to generate data and save it.
```python
# Import the libray as well as the corresponding simulator
import maifs.navierstokes as ns

# Simulator intialization (Simulation and fluid physical properties)
NS_sim = ns.Simulator(size      = 256,
			          density   = 1.,
			          viscosity = 1e-3,
				      reynolds  = 1e3,
			          predict   = "velocity",
			          forcing   = None)

# Generation of a trajectory for 500 timesteps (Fixed seed, random state)
traj_results = NS_sim.generate_trajectory(seed        = 69,
										  inner_steps = 20,
										  outer_steps = 500,
										  state       = None)

# Formatting the results into xArray and saving the trajectory
dataset_traj = sim.save_trajectory(traj_results, file_name = "sim_results")
```
where:
- **size** [int, 32 <]:  grid resolution;
- **density** [float, 0 <]:  density of the fluid;
- **viscosity** [float, 0 <] :  dynamic viscosity of the fluid
- **reynolds** [int, 0 <] : Reynolds number, i.e. must be set < 3000 for a laminar flow and 3000 < to have a turbulent flow.
- **predict** [str, "velocity", "vorticity"]: choose to solve the momentum equation for the *velocity* or for the *vorticity*
- **forcing** [str, "None", "kolmogorov", "taylor"]: choose to have no forcing *none*, a periodic forcing *kolmogorov* or a random forcing "taylor".
- **seed** [int, 0 <]: the seed used to fix random state, i.e. used to make experiments reproducible.
- **inner_steps** [int, 0 <]: the number of steps made by iterative solver to converge to the solution
- **outer_steps** [int, 0 <]: the number of simulation time steps
- **state** [2D matrix, "None", "State"]: initial state to start the simulation from. If set to none, it will start from a divergence free random state(seed) or from the one given.
- **file_name** [str] : the name of the file in which save the results.

<hr>
<p  align="center">
<b  style="font-size:30vw;">Navier-Stokes 2D - Going further </b>
</p>
<hr>

There are other utilities you can use such as:

-  Generate a (random) **state**:

```python
# 1 - If the simulator solves for the velocity, you can generate a state using
initial_state = NS_sim.random_state(seed              = 69,
							        compute_vorticity = False,
								    compute_FFT       = False)

# 2 - If the simulator solves for the vorticity, you need to give him:
# 		- Generate a velocity state
#		- Computes the vorticity from it using curl operator (compute_vorticity = True)
#		- Compute the FFT of this state by appliying jnp.fft.rfftn() to it (compute_FFT = True)
#
# 	The frequency representation of the vorticity is the state needed by the solver !
#
initial_state = NS_sim.random_state(seed              = 69,
							        compute_vorticity = True,
								    compute_FFT       = True)

# Note :
# You could be asking for a vorticity state, then correct it using and apply yourself the FFT.
```

- Generate a **single time step**:

```python
# From a given state, you can solve Navier-Stokes equation to obtain the next one.
next_state = NS_sim.generate_step(state = initial_state)

# Note :
# Be carefull with the state, velocity or frequency representation of vorticity
```

- Load a **saved dataset**:

```python
# Loading a dataset saved previously in H5 format
saved_dataset_traj = NS_sim.load_trajectory(file_name = "sim_results.h5")
```

- **Animate** your trajectory:

```python
# Once you have saved / load your xarray dataset, you can generate an animation !
#
# For a trajectory based on the velocity (horizontal here for example, v for vertical)
NS_sim.animate_trajectory(trajectory_data = dataset_traj.u,
						  file_name       = "dataset_u_animation",
						  fps             = 10)

# For a trajectory based on the vorticity
NS_sim.animate_trajectory(trajectory_data = dataset_traj.vorticity,
						  file_name       = "dataset_vorticity_animation",
						  fps             = 10)
```
