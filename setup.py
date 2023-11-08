import os
from setuptools import find_packages, setup

# General information
VERSION          = '1.0.0'
DESCRIPTION      = 'Montefiore AI - Fluid Simulator'
LONG_DESCRIPTION = 'Montefiore AI - Fluid Simulator : A library to make fluid simulations solving Burger (1D), Navier-Stokes (2D), and many more using JAXCFD !'

# All dependencies
requirement_path = "requirements.txt"
install_requires = []
if os.path.isfile(requirement_path):
    with open(requirement_path) as f:
        install_requires = f.read().splitlines()

# Seting up everything
setup(
      name             = "maifs",
      version          = VERSION,
      url              = 'https://github.com/VikVador/Montefiore-AI---FluidSimulator',
      author           = 'Victor Mangeleer',
      author_email     = 'vmangeleer@uliege.be',
      description      = DESCRIPTION,
      long_description = LONG_DESCRIPTION,
      packages         = ["maifs"],
      install_requires = install_requires,
)