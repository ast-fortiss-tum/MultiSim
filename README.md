# MultiSim - Replication Package for the Paper "Simulator Ensembles for Trustworthy Autonomous Driving Testing"
[ Content will be added soon! ]


This repo provides the MultiSim test generation approach, a testing technique which combines an ensemble of simulator during search-based testing to identify simulator-agnostic failures. This repo provides the integration of a LKAS case where a DNN model is tested in navigating a car on a road without obstacles. Integrated simulators are Donkey, Udacity and BeamNG. The framework builds upon [OpenSBT](https://github.com/opensbt/opensbt-core) 0.1.7.

<div align="center">
  <img src="/images/simulators_intro.png" height="300" style="background-color: rgb(44,46,57);"/>
</div>

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

To run a DSS experiment after completed MultiSim you can run the scripts:

```bash
TODO
```

## Running MultiSim with Prediction

To run MultiSim with a disagreement predictor, first you need to train the classifier.
You can train the classifier using this script (you can adopt the config in the top of the script, considering hyperparameters e.g.)


```bash
TODO
```

Now you can run a prediction based MultiSim search by assigning the predictor in the config.py file and
selecting the XZY Problem from the extended OpenSBT. As an example you should be able to run the experiment:

```bash
TODO
```


# Validation

To run validation for MultiSim, SingleSim or DSS run the following scripts (based on the simulator combination).

```bash
TODO
```

# Analysis

## RQ1. Effectiveness

To run the effectiveness analysis run the correspnding script  as follows  (only after validation is completed):

```bash
TODO
```

## RQ2. Efficiency

To run the efficiency analysis run the correspnding script  as follows (only after validation is completed):

```bash
TODO
```

# Results

The detailed results shown in the paper can be found in the folder [results](/results/) and seperated by RQ.


# Visualization

To generate the stacked plot results use the following script:

```bash
TODO
```

To run the statistical significance test use this script:

```bash
TODO
```

The boxplots for RQ3 can be created using the following command.

```bash
TODO
```




# Supplementary Material

You can find the supplementary material of our study in the folder [sup](sup/).
The folder is structured as follows:

- [sup/validation](sup/validation/): Contains the preliminary driving performance evaluation in all three simulators on 16 predefined roads.
- [sup/convergence](sup/convergence/): Contains Hypervolume results assessing the convergence of MultiSim and SingleSim.

## Validation

You can find all roads used for validation [here](sup/validation/roads/).
The detailed validation results are available in the respective simulator related folder.
The overall validation result with five 10 reexecutions is:

<div align="center">
  <img src="/sup/validation/preliminary_validation_u-d-b.png" height="400" style="background-color: rgb(44,46,57);"/>
</div>

## Convergence

Overview of Hypervolume over time.

<div align="center">
  <img src="/sup/convergence/subplots_combined_error_relative.png" height="400" style="background-color: rgb(44,46,57);"/>
</div>

# Authors

Lev Sorokin \
lev.sorokin@tum.de
