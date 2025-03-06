# Multi Simulation Testing Framework

**This repo is not official, so distribution without request is not permitted.**

This repo integrates the LKAS study where a DNN model is tested in navigating a car on a road without obstacles. Integrated simulators are donkey, udacity and BeamNG. The road generation for testing is guided by a SBT algorithm. The framework builds upon OpenSBT.

You can execute single simulation as well as multi simulation experiments.
In multi-simulation experiments the SUT is executed on multiple simulators for a given road, and the results are merged using a migration strategy.

To run the case study you have to Download the Donkey/Udacity simulator from this [link](https://drive.google.com/drive/folders/1e12fFeoqyd_IcheTL-48Nzp4pwOIQ4YE?usp=sharing). The BeamNG simulators is accessable upon request via [here](https://register.beamng.tech/). The DNN model (the SUT) can be downloaded from [here](https://drive.switch.ch/index.php/s/fMkAVQSCO5plOBZ?path=%2Flogs%2Fmodels). The best performing model considering previous research is the Dave2 model.

You have to update then the DNN_MODEL_PATH and DONKEY_EXE_PATH variable in the config.py file in this project.

# Installation

First, you need to install a Python 3.8 virtual environment in this project. After you have activated the environment, install the projects requirements by:

```bash
bash install.sh
```

# Test Installation

To test if the installation was successful run an experiment with the MockSimulator:

```bash
python run.py -e 8
```

A results folder should have been created with artefacts.

# Single Simulator Experiments

You can find predefined single sim experiments in single_experiments.py and execute a single sim experiment via its number, e.g:

```bash
python run.py -e 1000
```

Consider to specify the simulator in the simulate function of the experiment definition you want to use.

# Multi Simulator Experiments

You can find multi sim experiments in multi_experiments.py. One example experiment which uses Donkey and Udacity as an ensemble can be run via:

```bash
python run.py -e 62
```

Multi Simulator related fitness functions are defined in `fitness.migration`.

# Configuration

To configure thresholds such as the maximal speed of the SUT, criticality threshold, max simulation time or other parameters use `config.py`.

# Authors

Lev Sorokin \
lev.sorokin@tum.de \
sorokin@fortiss.org
