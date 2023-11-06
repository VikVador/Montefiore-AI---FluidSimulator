from setuptools import find_packages, setup

VERSION          = '1.0.0'
DESCRIPTION      = 'Montefiore AI - Fluid Simulator'
LONG_DESCRIPTION = 'Montefiore AI - Fluid Simulator : A library to make fluid simulations solving Burger (1D), Navier-Stokes (2D), and many more using JAXCFD !'

setup(
      name             = "maifs",
      version          = VERSION,
      url              = 'https://github.com/VikVador/Montefiore-AI---FluidSimulator',
      author           = 'Victor Mangeleer',
      author_email     = 'vmangeleer@uliege.be',
      description      = DESCRIPTION,
      long_description = LONG_DESCRIPTION,
      packages         = ["maifs",
                          "maifs.navierstokes"],
      install_requires = ["ipython==8.17.2",
                          "jax==0.4.11",
                          "jax_cfd==0.2.0",
                          "moviepy==1.0.3",
                          "numpy==1.24.3",
                          "setuptools==63.4.1",
                          "xarray==2022.9.0"]
)