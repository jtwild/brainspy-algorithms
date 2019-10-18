from setuptools import setup, find_packages

setup(name='brainspy-algorithms',
      version='0.0.0',
      description='Main optimisation algorithms used for training boron-doped silicon chips both in hardware and with surrogate models.',
      url='https://github.com/BraiNEdarwin/brainspy-algorithms',
      author='This has adopted part of the BRAINS skynet repository code, which has been cleaned and refactored. The maintainers of the code are Hans-Christian Ruiz Euler and Unai Alegre Ibarra.',
      author_email='u.alegre@utwente.nl',
      license='GPL-3.0',
      packages=find_packages(),
      install_requires=[
          'pyjson',
          'tqdm'
      ],
      python_requires='~=3.6',
      zip_safe=False)
