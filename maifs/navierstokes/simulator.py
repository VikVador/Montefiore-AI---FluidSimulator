# Librairies
import abc
import os
import xarray
import mgzip
import seaborn as sns
import dataclasses
import numpy as np
import matplotlib.pyplot as plt
import moviepy.video.io.ImageSequenceClip
from IPython.display import Video

import jax
from   jax_cfd.base import advection
from   jax_cfd.base import diffusion
from   jax_cfd.base import forcings
from   jax_cfd.base import grids
from   jax_cfd.base import pressure
from   jax_cfd.base import time_stepping
import jax.numpy        as jnp
import jax_cfd.base     as cfd
import jax_cfd.spectral as spectral
from   jax_cfd.data import visualization

class SimulatorNavierStokes():

    def __init__(self,
         size:      int   = 256,
         density:   float = 1.,
         viscosity: float = 1e-3,
         reynolds:  int   = 1e3,
         predict:   str   = "velocity",
         forcing:   str   = None
        ):
        super().__init__()

        # Security
        assert predict in ["velocity", "vorticity"],      "(ARGS - ERROR) Predict should be equal to 'velocity' or 'vorticity'"
        assert forcing in [None, "kolmogorov", "taylor"], "(ARGS - ERROR) Forcing should be equal to None, 'kolmogorov' or 'taylor'"

        # Properties
        self.size = size
        self.predict = predict
        self.forcing = forcing
        self.density = density
        self.reynolds = reynolds
        self.viscosity = viscosity

        # Domain length
        self.L = 2 * jnp.pi

        # Speed (roughly its maximum value to reach the desired flow regime, Re = density * speed * length / viscosity)
        self.v = (reynolds * viscosity)/(self.L * density)

        # Simulation domain (square)
        self.grid = grids.Grid(shape = (size, size), domain = ((0, self.L), (0, self.L)))

        # Timestep (must respect CFL ≈ dt/dx < 1, we arbitrarily set CFL to 0.1)
        self.dt = cfd.equations.stable_time_step(max_velocity = self.v, max_courant_number = 0.1, viscosity = viscosity, grid = self.grid)

        # Forcing (Example: atmospheric wind on the ocean surface)
        if forcing in ["kolmogorov", "taylor"]:
            self.forcing_fn = cfd.forcings.simple_turbulence_forcing(grid                = grid,
                                                                     constant_magnitude  = 1.0,
                                                                     constant_wavenumber = 4.0,
                                                                     linear_coefficient  = -0.1,
                                                                     forcing_type        = forcing)
        else:
            self.forcing_fn = lambda grid: forcings.no_forcing(grid = grid)

        # Navier-Stokes (1) - Solves for the velocity u
        if predict == "velocity":
            self.step_fn = cfd.equations.implicit_diffusion_navier_stokes(density   = density,
                                                                          viscosity = viscosity,
                                                                          grid      = grid,
                                                                          dt        = dt)
        # Navier-Stokes (2) - Solves for the vorticity q
        else:
            self.step_fn = spectral.time_stepping.imex_rk_sil3(
                                        spectral.equations.NavierStokes2D(viscosity  = viscosity,
                                                                          grid       = self.grid,
                                                                          drag       = 0 if forcing == None else 0.25,
                                                                          smooth     = True,
                                                                          forcing_fn = self.forcing_fn), self.dt)

    def random_state(self, seed:int, compute_vorticity:bool = False, compute_FFT:bool = False):

        # Divergence free random (or not) velocity state (∇.u ≈ 0 if incompressible flow)
        state = cfd.initial_conditions.filtered_velocity_field(rng_key          = jax.random.PRNGKey(seed),
                                                               grid             = self.grid,
                                                               maximum_velocity = self.v,
                                                               peak_wavenumber  = 4)

        # Computes the vorticity (Needed if Navier-Stokes solves for vorticity)
        state = cfd.finite_differences.curl_2d(state).data if compute_vorticity else state

        # Computes the fast fourier transform (Needed if Navier-Stokes solves for vorticity, it works on spectral data representation)
        return jnp.fft.rfftn(state) if compute_FFT else state

    def generate_step(self, state):

        # Computes one step of Navier-Stokes
        return self.step_fn(state)

    def generate_trajectory(self, seed:int, inner_steps:int, outer_steps:int, state = None,):

        # Determine if we need the vorticity or not
        need_vorticity = True if self.predict == "vorticity" else False

        # Initial state (random or given)
        state = self.random_state(seed = seed, compute_vorticity = need_vorticity, compute_FFT = need_vorticity) if state == None else state

        # Trajectory (Inner steps = steps to solve non-linearities | Outer steps = simulation steps)
        trajectory_fn = cfd.funcutils.trajectory(cfd.funcutils.repeated(self.step_fn, inner_steps), outer_steps)

        # Computes the trajectory
        _, trajectory = trajectory_fn(state)

        return trajectory

    def load_trajectory(self, file_name:str):
        return xarray.open_dataset(f"{file_name}.h5", engine = "h5netcdf")

    def save_trajectory(self, trajectory, file_name:str):

        # Trajectory information
        if self.predict == "velocity":

            # Data
            trajectory_data = {
                'u': (('time', 'x', 'y'), trajectory[0].data),
                'v': (('time', 'x', 'y'), trajectory[1].data),
            }

            # Number of time steps (needed for 'time' since we don't save outer and inner in init)
            samples = len(trajectory[0].data)

        else:

            # Transforming back into space-time domain
            trajectory = jnp.fft.irfftn(trajectory, axes=(1, 2))

            # Data
            trajectory_data = {
                'vorticity': (('time', 'x', 'y'), trajectory),
            }

            # Number of time steps
            samples = len(trajectory)

        # General information
        data_general = {

            # Flow
            'equation'   : "NavierStokes",
            'density'    : self.density,
            'reynols'    : self.reynolds,
            'viscosity'  : self.viscosity,
            'num_samples': samples,
            'base_res'   : self.size,
            'predict'    : self.predict,
            'forcing'    : self.forcing,
            'max_speed'  : self.v,
            'dt'   : self.dt,
            't_min': 0,
            't_max': samples * self.dt,
            'unit' : "s",
        }

        # Updating dataset with general information
        trajectory_data.update(data_general)

        # Creating the complete dataset
        dataset = xarray.Dataset(
            trajectory_data,
            coords = {
            'time': self.dt * jnp.arange(samples),
            'x'   : jnp.arange(self.grid.shape[0]) * 2 * jnp.pi / self.grid.shape[0],
            'y'   : jnp.arange(self.grid.shape[0]) * 2 * jnp.pi / self.grid.shape[0],
            }
        )

        # Saving the dataset
        dataset.to_netcdf(f"{file_name}.h5", mode = "w", engine = "h5netcdf")

        return dataset

    def animate_trajectory(self, trajectory_data:np.ndarray, file_name:str, fps:int = 10):

        # Name of the temporary folder to store pngs
        temp_f = "animation_temporary_folder"

        # Creates a temporary folder for the pngs
        if not os.path.exists(temp_f):
            os.mkdir(temp_f)

        # Creation of all the images
        images = visualization.trajectory_to_images(trajectory_data)

        # Saves all the images
        for t, image in enumerate(images):
            image.save(f"{temp_f}/timestep_{t}.png")

        # Loads all the data
        image_files = [f"{temp_f}/timestep_{t}.png" for t in range(len(images))]

        # Generates and saves the movie
        clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps = fps)
        clip.write_videofile(f'{file_name}.mp4')

        # Removes the files and directory
        for file in os.listdir(temp_f):
            if file.endswith('.png'):
                os.remove(f'{temp_f}/' + file)
