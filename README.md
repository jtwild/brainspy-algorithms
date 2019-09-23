# brainspy-algorithms
Main optimisation algorithms used for training boron-doped silicon chips both in hardware and with surrogate models.

[![Theory](https://img.shields.io/badge/brainspy--white.svg)](https://github.com/BraiNEdarwin/brainspy)
[![Theory](https://img.shields.io/badge/brainspy-instruments-lightblue.svg)](https://github.com/BraiNEdarwin/brainspy-instruments)
 [![Tools](https://img.shields.io/badge/brainspy-algorithms-blue.svg)](https://github.com/BraiNEdarwin/brainspy-algorithms)
[![Tools](https://img.shields.io/badge/brainspy-smg-darkblue.svg)](https://github.com/BraiNEdarwin/brainspy-smg)

## 1. Supported Algorithms ##
   * **Genetic Algorithm**: In computer science and operations research, a genetic algorithm (GA) is a meta-heuristic inspired by the process of natural selection that belongs to the larger class of evolutionary algorithms (EA). Genetic algorithms are commonly used to generate high-quality solutions to optimization and search problems by relying on bio-inspired operators such as mutation, crossover and selection. This algorithm is suitable for experiments with reservoir computing. [Mitchell, Melanie (1996). An Introduction to Genetic Algorithms. Cambridge, MA: MIT Press. ISBN 9780585030944.](https://books.google.nl/books?hl=en&lr=&id=0eznlz0TF-IC&oi=fnd&pg=PP9&ots=shoJ1029Jc&sig=wZ2khjtK5Gf468MmMZ-xOxepr1M&redir_esc=y#v=onepage&q&f=false)

   * **Gradient Descent**: Gradient descent is a first-order iterative optimization algorithm for finding the minimum of a function. To find a local minimum of a function using gradient descent, one takes steps proportional to the negative of the gradient (or approximate gradient) of the function at the current point. If, instead, one takes steps proportional to the positive of the gradient, one approaches a local maximum of that function; the procedure is then known as gradient ascent [D. P. Bertsekas (1997). Nonlinear programming. Journal of the Operational Research Society, Valume 48, Issue 3](https://doi.org/10.1057/palgrave.jors.2600425).


## 2. Installation instructions ##
* This repository uses the Python programming language, with [Anaconda](https://en.wikipedia.org/wiki/Anaconda_(Python_distribution)) as a package manager. In order to use this code, it is recommended to [download](https://www.anaconda.com/download) Anaconda (with python3 version) for your Operating System, and install it following the official instructions:
	* [Linux](https://docs.continuum.io/anaconda/install/linux/)
	* [Windows](https://docs.continuum.io/anaconda/install/windows/)
	* [Mac](https://docs.continuum.io/anaconda/install/mac-os/)
* The Anaconda package manager, is based on [environments](https://protostar.space/why-you-need-python-environments-and-how-to-manage-them-with-conda). In order to install the corresponding environment for the code hosted in this repository, follow these instructions:
	* [Clone](https://help.github.com/en/articles/cloning-a-repository) the repository into your computer.
	* Open the terminal in which anaconda is installed.
		* For Windows users, it might be installed as an independent terminal called Anaconda prompt.
		* For Mac and Linux users, it can be run from the regular terminal.
	* Inside the anaconda terminal, navigate to the main folder of the repository, in which the file [conda-env-conf.yml]() is, using the following commands:
		* *list directory* command: ```` ls````
		* *change directory* command: ```` cd my_folder````
  * Install the environment:````conda env create -f conda-env-conf.yml ````
	* Run the auto-installer: ````python autoinstall.py````
* Whenever developing or executing the code from this repository, the corresponding environment needs to be activated from the terminal where Anaconda is installed. This is done with the following command: ````conda activate bspyalgo````

* In order to finish, pytorch needs to be installed. Its installation has not been automated as it depends on your current CPU/GPU configuration. You can install it using the commands recommended in pytorch's [official website](https://pytorch.org/get-started/locally/).

## 4. Usage instructions ##
1) Go to the brainspy-algorithms/configs folder. If you are planning to use the genetic algorithm, get into the 'ga' folder. If you are planning to use the gradient descent algorithm get into the 'gd' folder. Make a copy of the configuration template and rename it. You can alter any configuration you need inside this new file.

2) You can now use the library as follows:
```` from bspyalgo import algorithm_manager as algo_mgr

     INPUTS = [[-1., 0.4, -1., 0.4, -0.8, 0.2], [-1., -1., 0.4, 0.4, 0., 0.], [-1., 1.4, -0.2, 0.1, -0.33, 0.2]]
     TARGETS = [1, 1, 0, 0, 1, 1]

     result = algo_mgr.get_algorithm('genetic','your_configs_filename').optimize(INPUTS, TARGETS)
````


## 5. Developer instructions ##
This code contains useful libraries that are expected to be used and maintained by different researchers and students at [University of Twente](https://www.utwente.nl/en/). If you would like to collaborate on the development of this or any projects of the [Brains Research Group](https://www.utwente.nl/en/brains/), you will be required to create a GitHub account. GitHub is based on the Git distributed version control system, if you are not familiar with Git, you can have a look at their [documentation](https://git-scm.com/).  If Git commands feel daunting, you are also welcome to use the graphical user interface provided by [GitHub Desktop](https://desktop.github.com/).

The development will follow the Github fork-and-pull workflow. You can have a look at how it works [here](https://reflectoring.io/github-fork-and-pull/).  Feel free to create your own fork of the repository. Make sure you [set this repository as upstream](https://help.github.com/en/articles/configuring-a-remote-for-a-fork), in order to be able to pull the latest changes of this repository to your own fork by [syncing](https://help.github.com/en/articles/syncing-a-fork). Pull requests on the original project will be accepted after maintainers have revised them. The code in this repository follows the [PEP8](https://www.python.org/dev/peps/pep-0008/) python coding style guide. Please make sure that your own fork is synced with this repository, and that it respects the PEP8 coding style.


## 3.1 Development environment
We recommend you to use the open source development environment of [Visual Studio Code](https://code.visualstudio.com/download) for python, which can be installed following the official [guide](https://code.visualstudio.com/docs/setup/setup-overview). For Ubuntu users, it is recommended to be installed using snap: ````sudo snap install --classic code````. We also recommend you to use an auto-formatter in order to follow PEP8. You can install several extensions that will help you with auto-formatting the code:

 * Open your conda terminal and activate the environment (if you do not have it activated already):  ````conda activate bspy-instr````
 * Install the auto-formatter packages from pip:
	 * ````pip install autopep8````
	 * ````pip install flake8````
 * From the same terminal, Open Visual Studio Code with the command: ````code````
 * Go to the extensions marketplace (Ctrl+Shift+X)  
 * Install the following extensions:  
	 * Python (Microsoft)  
	 * Python Extension Pack (Don Jayamanne)  
	 * Python Docs (Mukundan)  
	 * Python-autopep8 (himanoa)
	 * cornflakes-linter (kevinglasson)
 * On Visual Studio Code, press Ctrl+Shif+P and write "Open Settings (JSON)". The configuration file should look like this:
	 * Note: If you are using windows, you should add a line that points to your flake8 installation: ````"cornflakes.linter.executablePath": "C:/Users/Your_user/AppData/Local/Continuum/anaconda3/envs/Scripts/flake8.exe"````

````		
{
	"[python]": {  
	  "editor.tabSize": 4,  
	  "editor.insertSpaces": true,  
	  "editor.formatOnSave": true  
	},  
	"python.jediEnabled": true,  
	"editor.suggestSelection": "first",  
	"vsintellicode.modify.editor.suggestSelection": "automaticallyOverrodeDefaultValue"    
}
````

## Authors
The code is based on the instruments code of the [skynet](https://github.com/BraiNEdarwin/SkyNEt) legacy project. The authorship of this project is for those people who collaborated in it. It has been refactored and improved, and it is currently maintained by:
-   Hans-Christian Ruiz Euler [@hcruiz] [h.ruiz@utwente.nl](mailto:h.ruiz@utwente.nl)
-   Unai Alegre-Ibarra [@ualegre](https://github.com/ualegre) ([u.alegre@utwente.nl](mailto:u.alegre@utwente.nl))
-   Bram van de Ven, [@bbroo1](https://github.com/bbroo1) ([b.vandeven@utwente.nl](mailto:b.vandeven@utwente.nl))
