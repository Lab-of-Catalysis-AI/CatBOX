# Code for CatBOX

This script runs optimization path analysis experiments to compare the performance of different optimization algorithms on various problems. It records all X (input points) and y (function values) data throughout the optimization process.

![algorithm2](E:\science\BO\DDLBO\文章写作整理\figures\流程图\algorithm2.png)

## Features

- **Multiple Optimization Algorithms**: Supports SMKBO, CASMOPOLITAN, COCABO, MVRSM, TPE, Random Search, and GPyOpt
- **Various Problem Types**: 
  - Chemistry problems (OCM1（sperated）, OCM2（Combined）)
  - DAR (Drug Activity Response)
  - NO (Nitric Oxide)
  - Benchmark functions (Ackley, Rosenbrock, Schwefel, Griewank)
- **Comprehensive Data Recording**: Saves complete optimization paths including all evaluated points and their function values

## Requirements

Make sure you have installed all required dependencies in requirments.txt. 

And the custom modules:
- `mixed_test_func` (Chemistry, DAR, NO, benchmark functions)
- `runners` (optimization algorithm runners)

## Usage

### Basic Usage

```bash
python run_optimization_path.py
```

### Common Parameters

#### Problem Selection
- `-p, --problem`: Problem type (default: 'NO')
  - Options: 'OCM1', 'OCM2', 'DAR', 'NO', 'Ackley', 'Rosenbrock', 'Schwefel', 'Griewank'
- `-s, --sep`: Separation mode for Chemistry problems (default: 'sep')
  - Options: 'sep', 'atom', 'M1', 'M2', 'M3', 'Support', etc.

#### Benchmark Function Settings
- `--n_categorical`: Number of categorical variables (default: 3)
- `--n_continuous`: Number of continuous variables (default: 10)
- `--num_opts`: Number of options for categorical variables (default: 5)

#### Optimization Settings
- `--max_iters`: Maximum number of BO iterations (default: 150)
- `--n_init`: Number of initial random points (default: 20)
- `--n_trials`: Number of trials for the experiment (default: 1)
- `--seed`: Initial seed setting (default: 20)
- `--batch_size`: Batch size for BO (default: 1)
- `-a, --acq`: Acquisition function choice (default: 'ei')
  - Options: 'ucb', 'ei', 'thompson'

#### Algorithm Selection
- `--run_smk`: Run SMKBO (0 or 1, default: 0)
- `--run_cas`: Run CASMOPOLITAN (0 or 1, default: 0)
- `--run_cocabo`: Run COCABO (0 or 1, default: 0)
- `--run_mvrsm`: Run MVRSM (0 or 1, default: 0)
- `--run_tpe`: Run TPE (0 or 1, default: 0)
- `--run_rs`: Run Random Search (0 or 1, default: 0)
- `--run_gpyopt`: Run GPyOpt (0 or 1, default: 0)
- `--run_all`: Run all algorithms (0 or 1, default: 1)

#### SMKBO Specific Parameters
- `--num_mixtures1`: Number of Cauchy mixtures (default: 5)
- `--num_mixtures2`: Number of Gaussian mixtures (default: 4)
- `-cont_k, --continuous_kern_type`: Continuous kernel type (default: 'smk')
  - Options: 'mat52', 'rbf', 'smk'

#### COCABO Specific Parameters
- `-mix, --kernel_mix`: Mixture weight for production and summation kernel (default: 0.3)

#### Other Parameters
- `--save_path`: Save directory for log files (default: 'optimization_results_update')
- `--lamda`: Noise level to inject (default: 1e-6)
- `--ard`: Enable automatic relevance determination (flag)
- `--no_save`: Do not save results (flag)
- `--infer_noise_var`: Infer noise variance (flag)
- `--init_design`: Initialization design (default: 'random')
  - Options: 'random', 'best', 'non_zero'

## Examples

### Example 1: Run DAR problem with all algorithms

```bash
python run_optimization_path.py -p DAR --run_all 1 --max_iters 100 --n_trials 5
```

### Example 2: Run Ackley benchmark function with specific parameters

```bash
python run_optimization_path.py -p Ackley --n_categorical 3 --n_continuous 10 --num_opts 5 --run_gpyopt 1 --max_iters 150
```

### Example 3: Run Chemistry OCM2 problem with CASMOPOLITAN only

```bash
python run_optimization_path.py -p OCM2 -s sep --run_cas 1 --run_all 0 --max_iters 200 --n_init 30
```

### Example 4: Run multiple benchmark functions with custom settings

```bash
python run_optimization_path.py -p Rosenbrock --n_categorical 5 --n_continuous 20 --num_opts 10 --run_cocabo 1 --run_mvrsm 1 --run_all 0 --max_iters 150 --seed 42
```

### Example 5: Run with best initialization design

```bash
python run_optimization_path.py -p DAR --init_design best --run_all 1 --max_iters 150
```

## Output

The script generates the following outputs:

1. **Pickle Files**: Optimization path data saved as `.pkl` files
   - Filename format: `{problem}_{sep_mode}_{acq}_{trial}_path_analysis.pkl`
   - Contains tuples of (x_history, fx_history) for each algorithm
   - Saved in: `{save_path}/{problem}_{sep_mode}/`

2. **Runtime Summary**: CSV file with runtime information
   - Filename: `runtime_summary.csv`
   - Contains: trial number, algorithm name, separation mode, success status, duration, errors

### Output File Structure

Each pickle file contains a list of tuples, where each tuple represents one algorithm's results:
```python
[
    (smk_x_processed, smk_fx_processed),      # SMKBO C5G4
    (cas_x_processed, cas_fx_processed),      # CASMOPOLITAN
    (cocabo_x_processed, cocabo_fx_processed), # COCABO
    (mvrsm_x_processed, mvrsm_fx_processed),  # MVRSM
    (tpe_x_processed, tpe_fx_processed),      # TPE
    (gpyopt_x_processed, gpyopt_fx_processed), # GPyOpt
    (rs_x_processed, rs_fx_processed)         # Random Search
]
```

Each `x` array has shape `(max_iters, problem_dim)` and each `fx` array has shape `(max_iters, 1)`.

## Notes

- The script automatically handles memory cleanup after each problem to prevent memory accumulation
- Each trial uses a different seed (seed + trial_number) to ensure reproducibility while maintaining diversity
- If an old result file exists, the script will attempt to merge new results with old ones
- For mixed-type problems, categorical variables are stored as strings and continuous variables as numbers
- The script supports resuming interrupted runs by merging with existing pickle files

## Troubleshooting

1. **Import Errors**: Make sure all required modules are in your Python path
2. **Memory Issues**: The script includes automatic memory cleanup, but for very long runs, you may need to run fewer trials at a time
3. **File Not Found**: Ensure that model files for Chemistry/DAR/NO problems are in the correct directories
4. **Algorithm Failures**: Check the runtime_summary.csv file for detailed error messages

## License

[Add your license information here]
