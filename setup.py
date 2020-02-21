from setuptools import setup, find_packages

setup(name='brainspy-algorithms',
      version='0.0.0',
      description='Main optimisation algorithms used for training boron-doped silicon chips both in hardware and with surrogate models.',
      url='https://github.com/BraiNEdarwin/brainspy-algorithms',
      author='This has adopted part of the BRAINS skynet repository code, which has been cleaned and refactored. The maintainers of the code are Unai Alegre Ibarra and Hans-Christian Ruiz Euler.',
      author_email='u.alegre@utwente.nl',
      license='GPL-3.0',
      packages=find_packages(),
      install_requires=[
          'pyjson',
          'tqdm',
          'torch-optimizer'
      ],
      python_requires='~=3.8.1',
      zip_safe=False)
