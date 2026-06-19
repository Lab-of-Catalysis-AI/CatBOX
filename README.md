# Code for CatBOX

This repository contains the code and pretrained surrogate models used for CatBOX optimization path analysis experiments. It records both the queried inputs `X` and objective values `y` across the optimization process so results can be reproduced and compared across algorithms.

Website for CatBOX: [CatBOX](https://catbox.top)

![algorithm overview](./algorithm5.png)

## Reproducibility Checklist

To reproduce the included experiments reliably:

1. Use Python `3.10` or `3.11`.
2. Install Git LFS before cloning or run `git lfs pull` after cloning.
3. Install the pinned packages from `requirements.txt`.
4. Run `python optimization_path_analysis/run_optimization_path.py --help` first to confirm the environment is healthy.

The current dependency stack was not validated on Python `3.12`, and some packages may fail there even before an experiment starts.
The shipped AutoGluon predictors were created with AutoGluon `1.2` and Python `3.10`; loading them with newer AutoGluon releases may fail at inference time.

## Repository Contents

- `optimization_path_analysis/`: experiment entrypoint and algorithm runners
- `mixed_test_func/`: benchmark problems and surrogate-based chemistry/SCR/DAR objectives
- `cas/`, `cocabo/`, `mvrsm/`: optimization backends
- `optimization_results_update/`: default output directory for generated results

## Installation

```bash
git lfs install
git clone <your-repo-url>
cd package_to_GitHub
git lfs pull
python -m pip install -r requirements.txt
```

## Usage

### Show all options

```bash
python optimization_path_analysis/run_optimization_path.py --help
```

### Important behavior

- `--run_all` defaults to `1`.
- If you only want selected algorithms, set `--run_all 0` and then enable the needed `--run_*` flags.
- Supported real/surrogate problems: `OCM`, `DAR`, `SCR`
- Supported benchmark problems: `Ackley`, `Rosenbrock`, `Schwefel`, `Griewank`

### Common Parameters

#### Problem selection

- `-p, --problem`: problem name
- `-s, --sep`: separation mode for SCR style problems
- `--init_design`: initialization design, usually `random` or `best`

#### Benchmark settings

- `--n_categorical`: number of categorical variables
- `--n_continuous`: number of continuous variables
- `--num_opts`: number of options for each categorical variable

#### Optimization settings

- `--max_iters`: total optimization iterations
- `--n_init`: number of initial random points
- `--n_trials`: number of repeated trials
- `--seed`: base random seed
- `--batch_size`: batch size
- `-a, --acq`: acquisition choice, one of `ucb`, `ei`, `thompson`

#### Algorithm flags

- `--run_smk`
- `--run_cas`
- `--run_cocabo`
- `--run_mvrsm`
- `--run_tpe`
- `--run_rs`
- `--run_gpyopt`

## Example Commands

### SCR with all algorithms

```bash
python optimization_path_analysis/run_optimization_path.py -p SCR --run_all 1 --max_iters 100 --n_trials 5
```

### OCM with CASMOPOLITAN only

```bash
python optimization_path_analysis/run_optimization_path.py -p OCM --run_all 0 --run_cas 1 --max_iters 200 --n_init 30
```

Running `-p OCM` now automatically uses the shipped `OCM2_all_update_true` surrogate model.

### DAR with best initialization

```bash
python optimization_path_analysis/run_optimization_path.py -p DAR --init_design best --run_all 1 --max_iters 150
```

Running `-p DAR` now automatically uses the shipped `DAR_medium` surrogate model in `normal` mode.

### Ackley benchmark with GPyOpt only

```bash
python optimization_path_analysis/run_optimization_path.py -p Ackley --run_all 0 --run_gpyopt 1 --n_categorical 3 --n_continuous 10 --num_opts 5 --max_iters 150
```

## Outputs

The script writes:

1. Optimization path `.pkl` files
2. A `runtime_summary.csv` file for the current save directory

### Output layout

- Real/surrogate problems: `{save_path}/{problem}_{sep}/`
- Benchmark problems: `{save_path}/benchmarks{n_categorical}+{n_continuous}+{num_opts}/{problem}/`

### Pickle structure

Each saved pickle contains a list of tuples:

```python
[
    (smk_x_processed, smk_fx_processed),
    (cas_x_processed, cas_fx_processed),
    (cocabo_x_processed, cocabo_fx_processed),
    (mvrsm_x_processed, mvrsm_fx_processed),
    (tpe_x_processed, tpe_fx_processed),
    (gpyopt_x_processed, gpyopt_fx_processed),
    (rs_x_processed, rs_fx_processed),
]
```

Each `x` array has shape `(max_iters, problem_dim)` and each `fx` array has shape `(max_iters, 1)`.
