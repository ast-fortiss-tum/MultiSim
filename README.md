# MultiSim - Replication Package for the Paper "Simulator Ensembles for Trustworthy Autonomous Driving Testing"
[ Content will be added soon! ]


This repo provides the MultiSim test generation approach, a testing technique which combines an ensemble of simulator during search-based testing to identify simulator-agnostic failures. This repo provides the integration of a LKAS case where a DNN model is tested in navigating a car on a road without obstacles. Integrated simulators are Donkey, Udacity and BeamNG. The framework builds upon [OpenSBT](https://github.com/opensbt/opensbt-core) 0.1.7.

# Installation

To run the case study you have to Download the Donkey/Udacity simulator from this [link](https://drive.google.com/drive/folders/1e12fFeoqyd_IcheTL-48Nzp4pwOIQ4YE?usp=sharing). The BeamNG simulators is accessable upon request via [here](https://register.beamng.tech/). The DNN model (the SUT) can be downloaded from [here](https://drive.switch.ch/index.php/s/fMkAVQSCO5plOBZ?path=%2Flogs%2Fmodels). The best performing model considering previous research is the Dave2 model.

You have to update then the DNN_MODEL_PATH and DONKEY_EXE_PATH variable in the config.py file in this project.

First, you need to install a Python 3.8 virtual environment in this project. After you have activated the environment, install the projects requirements by:

```bash
bash install.sh
```
To test if the installation was successful run an experiment with the MockSimulator:

```bash
python run.py -e 8
```

A results folder should have been created with artefacts.

# Search Execution 
## Configuration

To configure thresholds such as the maximal speed of the SUT, criticality threshold, max simulation time or other parameters use `config.py`.

## Running MultiSim Experiments

You can find multi sim experiments in multi_experiments.py. One example experiment which uses Donkey and Udacity as an ensemble can be run via:

```bash
python run.py -e 62
```

Multi Simulator related fitness functions are defined in `fitness.migration`.


## Running SingleSim Experiments
You can find predefined single sim experiments in single_experiments.py and execute a single sim experiment via its number, e.g:

```bash
python run.py -e 1000
```

Consider to specify the simulator in the simulate function of the experiment definition you want to use.

## Running DSS Experiments

TODO
# Validation

# Analysis

# Results and Supplementary Material

# Authors

Lev Sorokin \
lev.sorokin@tum.de
