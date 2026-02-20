# MultiSim - Replication Package for the Paper "Simulator Ensembles for Trustworthy Autonomous Driving Systems Testing"

⚠️ **Note:** More content will be added soon!

This repo provides the MultiSim test generation approach, a testing technique that combines an ensemble of simulators during search-based testing to identify simulator-agnostic failures. This repo provides the integration of an LKAS case study where a DNN model is tested in navigating a car on a road without obstacles. Integrated simulators are Donkey, Udacity, and BeamNG. The framework builds upon [OpenSBT](https://github.com/opensbt/opensbt-core) 0.1.7. Some parts of the code are taken and adapted from the [Digital Siblings approach](https://github.com/testingautomated-usi/maxitwo).

<img align="center" src="/images/simulators_intro.png" height="200"/>

## Table of Contents

- [Installation](#installation)  
- [Search Execution](#search-execution)  
- [Validation](#validation)  
- [Analysis](#analysis)  
- [Results](#results)  
- [Visualization](#visualization)  
- [Supplementary Material](#supplementary-material)  
- [Authors](#authors)  

# Installation

To run the case study you have to Download the Donkey/Udacity simulator from this [link](https://drive.google.com/drive/folders/1e12fFeoqyd_IcheTL-48Nzp4pwOIQ4YE?usp=sharing). The BeamNG simulator is accessible upon request via [here](https://register.beamng.tech/). The DNN model (the SUT) can be downloaded from [here](https://drive.switch.ch/index.php/s/fMkAVQSCO5plOBZ?path=%2Flogs%2Fmodels). The best-performing model considering previous research is the Dave2 model.

The code is in the folder called `multisim`. All following operations have to be performed within this folder.

You have to update then the DNN_MODEL_PATH and DONKEY_EXE_PATH variable in the config.py file in this project.

First, you need to install a Python 3.8 virtual environment in this project. After you have activated the environment, install the project requirements:

```bash
bash install.sh
```
To test if the installation was successful run an experiment with the MockSimulator:

```bash
python run.py -e 8
```

A results folder should have been created with artifacts.

# Search Execution 
## Configuration

To configure thresholds such as the maximal speed of the SUT, criticality threshold, max simulation time or other parameters use `config.py`.

## Running MultiSim Experiments

You can find multi-sim experiments in multi_experiments.py. One example experiment that uses Donkey and Udacity as an ensemble can be run via:

```bash
python run.py -e 64
```

Multi Simulator-related fitness functions are defined in `fitness.migration`.


## Running SingleSim Experiments
You can find predefined single sim experiments in single_experiments.py and execute a single sim experiment via its number, e.g:

```bash
python run.py -e 1000
```

Consider specifying the simulator in the simulate function of the experiment definition you want to use.

## Running DSS Experiments

To run a DSS experiment after completed MultiSim execution you can run for instance the following script, which execute tests found by BeamNG (b) in Donkey (d), and vice versa, followed by union of resulting feature maps.

```bash
bash dsim/run_migrate_and_simulate_bd.sh
```
Related scripts that run DSS for the combinations BeamNG + Udacity, and Udacity and Donkey with DSS are available in the same folder.

## Running MultiSim with Prediction

To run MultiSim with a disagreement predictor, first, you need to train the classifier.
You can train the classifier using this script (you can adopt the config in the top of the script, considering hyperparameters e.g.):


```bash
python -m prediction.disagree_predict
```

Now you can run a prediction-based MultiSim search by assigning the predictor in the config.py (DISAGREE_CLASSIFIER_1) file and
selecting the `ADSMultiSimAgreementProblemDiversePredict` Problem from the extended OpenSBT problem. As an example, you should be able to run the experiment:

```bash
python run.py -e 67
```


# Validation

To run in addition validation for MultiSim, SingleSim you can use the flag `-v` when calling the run function, or isolate from search by executing the following script (which execution 5 times each failures in the simulators Udacity and Donkey):

```python
python -m scripts_analysis.validation
    --n_repeat_validation 5 \
    --write_results_extended \
    --save_folder "results/<path>" \
    --simulators "udacity" "donkey" \
    --percentage_validation 100 \
    --folder_name_combined_prefix "validation_combined_" \
    --folder_validation_prefix "validation_" \
    --do_combined \
    --only_failing_cells
```

# Analysis

## Exhaustive Testing

To execute multple experiments in sequence the following script can be used. Inside the script the experiment number (whether SingleSim, MulitSim or DSS related) with search parameters and seeds can be defined. The results will be stored seed-wise in the results folder with the given name.

```bash
bash scripts_analysis/run_exp_multi.sh
```
## Validation

To run validation for DB in Udacity run the following for instance (inside the scripts the path to the results might need to be adopted):

```bash
bash scripts_analysis/run_validation_db_u.sh 
```

To run validation for DSS using BeamNG and Donkey you need for instance to execute the following script (inside the script the path to the results might need to be adopted):

```bash
bash scripts_analysis/dss_all/run_validation_dsim_u.sh 
```


## RQ1. Effectiveness

To run the effectiveness analysis after the completed search run the corresponding script  as follows  (only after validation is completed, sim_path indicates the results subfolder (msim or sim_1)):

```python
python scripts_analysis.ov.generate_overview_runs --folder_path "results/<path_to_results>" \ 
                 --combo "bd_u" \
                 --sim_path "sim_1" \
                 --save_folder "<output_path>
                 --sim "udacity" \
                 --seeds 120 123 \
                 --metrics_names 'valid_rate' 'n_valid'
```

The script calculate an overview over 2 runs for instance for the metrics valid_rate and n_valid for the stored results.

## RQ2. Efficiency

To run the efficiency analysis run the dedicated script as follows (only after validation is completed):

```python
python -m scripts_analysis.get_first_last_valid_new_time \
  --paths "../msim-results/analysis/analysis_runs_20-12-2024_BD/" \
  --validation_folders "validation_combined_count-3_u" \
  --sim_map "msim" \
  --sim_names "bd" \
  --save_folder_all "../msim-results/analysis/analysis_runs_20-12-2024_BD//" \
  --seeds 29 120 348 655 684 702 760 794 859 904 \
  --n_runs 10
```

In this example, the MultiSim BD combination is evaluated on 10 executed runs for instance.

# Results

The detailed results shown in the paper can be found in the folder [results](/results/) and separated by RQ.


# Visualization

To generate the stacked plot results use the following command:

```python
python -m scripts_analysis.stack.generate_stacked_plot \
                            --folder_paths  "../msim-results/analysis/analysis_runs_20-12-2024_BD/" \
                                            "../msim-results/analysis/analysis_runs_26-12-2024_BU/" \
                                            "../msim-results/analysis/analysis_runs_08-01-2025_UD/" \
                            --combos "u" "d" "b" \
                            --sim_paths "msim" "msim" "msim"\
                            --sims "BD" "BU" "UD" \
                            --plot_names "BD" "BU" "UD"\
                            --n_runs 3 \
                            --seeds  702 794 655\
                            --save_folder "../msim-results/analysis/analysis_runs_01-2025/stacked_plot/" \
                            --is_dss 0 0 0
```

The script will generate in this example a stacked plot visualizing the number of failures and valid failures for three compared approaches.

To run the statistical significance test use this script: The paths to the results need to be indicated in the script.

```bash
python -m scripts_analysis.evaluate_stat_sign
```

# Supplementary Material

You can find the supplementary material of our study in the folder [sup](sup/).
The folder is structured as follows:

- [sup/validation](sup/validation/): Contains the preliminary driving performance evaluation in all three simulators on 16 predefined roads.
- [sup/convergence](sup/convergence/): Contains Hypervolume results assessing the convergence of MultiSim and SingleSim.

## Validation

You can find all roads used for validation [here](sup/validation/roads/).
The detailed validation results are available in the respective simulator-related folder.
The overall validation result with 10 re-executions is:

<div align="center">
  <img src="/sup/validation/preliminary_validation_u-d-b.png" height="400" style="background-color: rgb(44,46,57);"/>
</div>

## Convergence

Overview of Hypervolume over time, including the average value and standard deviation results over 10 search executions. 

<div align="center">
  <img src="/sup/convergence/subplots_combined_error_relative.png" height="400" style="background-color: rgb(44,46,57);"/>
</div>

## Citation

A preprint of the paper can be found on [arXiv](https://arxiv.org/abs/2503.08936).

If you use our work in your research, or it helps it, or if you simply like it, please cite it in your publications. 
Here is an example BibTeX entry:

```
@article{sorokin2026multisim,
  author    = {Lev Sorokin and Matteo Biagiola and Andrea Stocco},
  title     = {Simulator ensembles for trustworthy autonomous driving systems testing},
  journal   = {Empirical Software Engineering},
  year      = {2026},
  volume    = {31},
  number    = {4},
  pages     = {80},
  doi       = {10.1007/s10664-026-10821-7}
}
```

# Authors

<table>
  <tr>
    <td><a href="https://github.com/leviathan321">
      <img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" width="30" height="30">
    </a></td>
    <td>Lev Sorokin | lev.sorokin@tum.de</td>
  </tr>
</table>
