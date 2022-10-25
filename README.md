# Electronic Density of States and Energy Bands Plotter 

The script `dosbandsplot.py`, written in Python 3, can represent electronic bands and densities of states calculated by the software Quantum ESPRESSO. The script needs the modules `os`, `sys`, `ast`, `copy`, `glob`, `pathlib` and `configparser`, from the Python’s standard library, and also the libraries NumPy and Matplotlib.

In order to run the code for a file of bands or densities of states (or both), it is necessary to specify some parameters in a configuration file. For running the code one has to write the following in the command line, being in the folder of the file dosbands.py, `python3 dosbandsplot.py <path>`. The argument `<path>` is the path of the configuration file. The configuration file is a plain text document with the writing formating style defined by the `configparser` module. Here one can specify the parameters of the script `dosbandsplot.py` by defining different variables. These variables are divided in four distinct blocks: the preamble, the common options, the bands mode options, and the density of states mode options.

You can read the user guide to learn how to use this script, and you can try the example using the data and the configuration file contained in the `example` folder.

Quantum ESPRESSO website: https://www.quantum-espresso.org/
