import numpy as np
import argparse
import os
import pickle
import sys
import time
import pandas as pd
from tqdm import tqdm
import gc
import torch

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mixed_test_func import Chemistry,DAR,NO,Ackley_benchmark,Rosenbrock_benchmark,Schwefel_benchmark,Griewank_benchmark

from runners import run_cas, run_cocabo, run_mvrsm, run_tpe, run_random_search, run_gpyopt


def clear_memory():
    """
    Clear memory to prevent issues caused by memory accumulation
    """  
    try:
        # Clear PyTorch cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # Force garbage collection
        gc.collect()
        print("Memory cleared successfully")
    except Exception as e:
        print(f"Warning: Memory cleanup failed: {e}")


def run_optimization_path_analysis():
    """
    Run optimization path analysis experiment
    Compare performance of different optimization algorithms, record all X and y data
    """
    parser = argparse.ArgumentParser('Run Optimization Path Analysis')

    # Chemistry-specific setting
    parser.add_argument('-p', '--problem', type=str, default='NO')
    parser.add_argument('-s', '--sep', type=str, default='sep', help='Catalyst separated or combined.')
    parser.add_argument('--init_design', type=str, default='random', help='**random** initialization or existing **best** points')

    # Benchmark function settings
    parser.add_argument('--n_categorical', type=int, default=3, help='Number of categorical variables for benchmark functions')
    parser.add_argument('--n_continuous', type=int, default=10, help='Number of continuous variables for benchmark functions')
    parser.add_argument('--num_opts', type=int, default=5, help='Number of options for categorical variables')

    # General parameters
    parser.add_argument('--seed', type=int, default=20, help='**initial** seed setting')
    parser.add_argument('--save_path', type=str, default=f'optimization_results_update', help='save directory of the log files')
    parser.add_argument('--max_iters', type=int, default=150, help='Maximum number of BO iterations 150.')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size for BO.')
    parser.add_argument('--n_trials', type=int, default=1, help='number of trials for the experiment')
    parser.add_argument('--n_init', type=int, default=20, help='number of initialising random points')
    parser.add_argument('-a', '--acq', type=str, default='ei', help='choice of the acquisition function.')

    parser.add_argument('--lamda', type=float, default=1e-6, help='the noise to inject for some problems')
    parser.add_argument('--ard', action='store_true', help='whether to enable automatic relevance determination')
    parser.add_argument('--random_seed_objective', type=int, default=20, help='The default value of 20 is provided also in COMBO')
    parser.add_argument('--no_save', action='store_true', help='If activated, do not save the current run into a log folder.')
    parser.add_argument('--infer_noise_var', action='store_true')

    # SMKBO
    parser.add_argument('--run_smk', type=int, default=0, help='Run SMKBO or not')
    parser.add_argument('-k', '--kernel_type', type=str, default=None, help='specifies the kernel type')
    parser.add_argument('-cont_k', '--continuous_kern_type', type=str, default='smk', help='specifies the continuous kernel type (mat52, rbf, smk)')
    parser.add_argument('--num_mixtures1', type=int, default=5, help='Number of **Cauchy** mixtures')
    parser.add_argument('--num_mixtures2', type=int, default=4, help='Number of **Gaussian** mixtures')

    # CASMOPOLITAN
    parser.add_argument('--run_cas', type=int, default=0, help='Run CASMOPOLITAN or not')

    # COCABO
    parser.add_argument('--run_cocabo', type=int, default=0, help='Run COCABO or not')
    parser.add_argument('-mix', '--kernel_mix',
                            help='Mixture weight for production and summation kernel. Default = 0.0', default=0.3,
                            type=float)

    # MVRSM
    parser.add_argument('--run_mvrsm', type=int, default=0, help='Run MVRSM or not')

    # TPE
    parser.add_argument('--run_tpe', type=int, default=0, help='Run TPE or not')

    # Random Search
    parser.add_argument('--run_rs', type=int, default=0, help='Run random search or not')

    # GPyOpt
    parser.add_argument('--run_gpyopt', type=int, default=0, help='Run GPyOpt or not')

    # Run all algorithms
    parser.add_argument('--run_all',type=int, default=1, help='Run all algorithms (SMKBO, CAS, COCABO, MVRSM, TPE, RS, EDBO, GPyOpt)')

    args = parser.parse_args()
    
    # If --run_all is specified, set all algorithms to run
    if args.run_all:
        args.run_smk = 0
        args.run_cas = 0
        args.run_cocabo = 0
        args.run_mvrsm = 0
        args.run_tpe = 0
        args.run_rs = 0
        args.run_gpyopt = 1
        print("Running all algorithms: SMKBO, CAS, COCABO, MVRSM, TPE, RS, EDBO, GPyOpt")
    options = vars(args)
    print("Optimization Path Analysis Configuration:")
    print(options)

    # Sanity checks
    assert args.acq in ['ucb', 'ei', 'thompson'], 'Unknown acquisition function choice ' + str(args.acq)

    # Save files
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir_ori = os.path.join(cur_dir, args.save_path)

    sep_modes = ['normal']
    #problems = ['Ackley','Dejong', 'Michalewicz', 'Rosenbrock','Levy','Rastrigin']
    problems = ['DAR']
    
    # Define different parameter combinations: (n_categorical, n_continuous, num_opts)
    param_combinations = [
        (3, 20, 5),   # 3+20+5
        (3, 30, 5),   # 3+30+5
        (5, 30, 5),   # 5+30+5
        (10, 30, 5),  # 10+30+5
    ]
    
    # for n_cat, n_cont, n_opts in param_combinations:
    #     args.n_categorical = n_cat
    #     args.n_continuous = n_cont
    #     args.num_opts = n_opts
    #     print(f'\n{"="*80}')
    #     print(f'Running with parameters: n_categorical={n_cat}, n_continuous={n_cont}, num_opts={n_opts}')
    #     print(f'{"="*80}')
        
    #     # Create benchmark directory for current parameter combination (without modifying original save_dir_ori)
    #     # For benchmark mode, base directory is benchmarks{n_cat}+{n_cont}+{n_opts}
    benchmark_base_dir = os.path.join(save_dir_ori, f'benchmarks{args.n_categorical}+{args.n_continuous}+{args.num_opts}')
        
    for problem in problems:
        args.problem = problem
        for sep_mode in sep_modes:
            print(sep_mode + ' is running')
            save_mode = args.problem + '_' + sep_mode
            
            if sep_mode == 'benchmark':
                save_dir = os.path.join(benchmark_base_dir, save_mode)
            else:
                save_dir = os.path.join(save_dir_ori, save_mode)

            if not os.path.exists(save_dir):
                print('save_dir ',save_dir)
                os.makedirs(save_dir)
            print(f'\n{"="*60}')
            print(f'Starting optimization path analysis for {sep_mode.upper()} mode')
            print(f'{"="*60}')
            
            # Set separation mode
            args.sep = sep_mode
            runtime_records = []
            
            head = args.problem
            if args.init_design == 'best':
                head += '-b'
            else:
                head += '-r'
            head += '-' + args.sep

            for t in tqdm(range(args.n_trials), desc=f"Optimization Path Analysis Trials ({sep_mode})", unit="trial"):
                t = t + 2
                print(f'\n----- Starting Optimization Path Analysis Trial {t + 1} / {args.n_trials} -----')
                name = f"{head}_{args.acq}_{t}_path_analysis.pkl"
                #name = 'test.pkl'
                filename = os.path.join(save_dir, name)

                # Set different seed for each trial to ensure reproducibility but different results across trials
                trial_seed = args.seed + t if args.seed is not None else t
            

                if args.problem == 'OCM2' or args.problem == 'OCM1':
                    f = Chemistry(normalize=False, lamda=args.lamda, seed=trial_seed, sep=args.sep, prob=args.problem)
                elif args.problem == 'DAR':
                    f = DAR(normalize=False, lamda=args.lamda, seed=trial_seed, sep=args.sep)
                elif args.problem == 'NO':
                    f = NO(normalize=False, lamda=args.lamda, seed=trial_seed, sep=args.sep)
                elif args.problem == 'Ackley':

                    f = Ackley_benchmark(n_categorical=args.n_categorical, n_continuous=args.n_continuous,
                                    num_opts=args.num_opts, normalize=False, seed=trial_seed)
                elif args.problem == 'Rosenbrock':
                    f = Rosenbrock_benchmark(n_categorical=args.n_categorical, n_continuous=args.n_continuous,
                                    num_opts=args.num_opts, normalize=False, seed=trial_seed)
                elif args.problem == 'Schwefel':
                    f = Schwefel_benchmark(n_categorical=args.n_categorical, n_continuous=args.n_continuous,
                                    num_opts=args.num_opts, normalize=False, seed=trial_seed)
                elif args.problem == 'Griewank':
                    f = Griewank_benchmark(n_categorical=args.n_categorical, n_continuous=args.n_continuous,
                                    num_opts=args.num_opts, normalize=False, seed=trial_seed)
                else:
                    raise ValueError('Unrecognised problem type %s' % args.problem)

                # Initialize fx and x arrays as empty lists to store results from each iteration
                smk_fx_c5g4, smk_x_c5g4 = [], []  # C5G4
                cas_fx, cas_x = [], []
                cocabo_fx, cocabo_x = [], []
                mvrsm_fx, mvrsm_x = [], []
                tpe_fx, tpe_x = [], []
                rs_fx, rs_x = [], []
                gpyopt_fx, gpyopt_x = [], []

                if args.run_smk:
                    args.continuous_kern_type = 'smk'

                    start_time = time.time()
                    try:
                        # C5G4: Cauchy=5, Gaussian=4
                        args.num_mixtures1, args.num_mixtures2 = 5, 4
                        args.seed = trial_seed  # Set different seed for each trial
                        print(f"Running SMKBO (C5G4) Cauchy {args.num_mixtures1}, Gaussian {args.num_mixtures2}......")
                        smk_x_c5g4, smk_fx_c5g4 = run_cas(f, args)
                        print(f"SMKBO (C5G4) completed with {len(smk_fx_c5g4)} iterations")
                        duration = time.time() - start_time
                        runtime_records.append({
                            'trial': t,
                            'algorithm': 'SMKBO_C5G4',
                            'sep_mode': args.sep,
                            'success': True,
                            'duration_sec': duration
                        })
                    except Exception as e:
                        duration = time.time() - start_time
                        error_msg = str(e)
                        runtime_records.append({
                            'trial': t,
                            'algorithm': 'SMKBO_C5G4',
                            'sep_mode': args.sep,
                            'success': False,
                            'duration_sec': duration,
                            'error': error_msg
                        })
                        print(f"Error running SMKBO C5G4: {error_msg}")
                        import traceback
                        print(f"Full traceback:")
                        traceback.print_exc()

                if args.run_cas:
                    start_time = time.time()
                    try:
                        print("Running CASMOPOLITAN with Matern52 Kernel......")
                        args.continuous_kern_type = 'mat52'
                        args.seed = trial_seed  # Set different seed for each trial
                        cas_x, cas_fx = run_cas(f, args)
                        print(f"CASMOPOLITAN completed with {len(cas_fx)} iterations")
                        duration = time.time() - start_time
                        runtime_records.append({
                            'trial': t,
                            'algorithm': 'CASMOPOLITAN',
                            'sep_mode': args.sep,
                            'success': True,
                            'duration_sec': duration
                        })
                    except Exception as e:
                        duration = time.time() - start_time
                        runtime_records.append({
                            'trial': t,
                            'algorithm': 'CASMOPOLITAN',
                            'sep_mode': args.sep,
                            'success': False,
                            'duration_sec': duration,
                            'error': str(e)
                        })
                        print(f"Error running CAS: {e}")

                if args.run_cocabo:
                    start_time = time.time()
                    try:
                        print("Running COCABO......")
                        cocabo_x, cocabo_fx = run_cocabo(f=f, budget=args.max_iters, initN=args.n_init,
                                        kernel_mix=args.kernel_mix, n_trial=t, seed=trial_seed, args=args)
                        print(f"COCABO completed with {len(cocabo_fx)} iterations")
                        duration = time.time() - start_time
                        runtime_records.append({
                            'trial': t,
                            'algorithm': 'COCABO',
                            'sep_mode': args.sep,
                            'success': True,
                            'duration_sec': duration
                        })
                    except Exception as e:
                        duration = time.time() - start_time
                        runtime_records.append({
                            'trial': t,
                            'algorithm': 'COCABO',
                            'sep_mode': args.sep,
                            'success': False,
                            'duration_sec': duration,
                            'error': str(e)
                        })
                        print(f"Error running COCABO: {e}")

                if args.run_mvrsm:
                    start_time = time.time()
                    try:
                        print("Running MVRSM......")
                        mvrsm_x, mvrsm_fx = run_mvrsm(f=f, max_evals=args.max_iters, rand_evals=args.n_init, seed=trial_seed, args=args)
                        print(f"MVRSM completed with {len(mvrsm_fx)} iterations")
                        duration = time.time() - start_time
                        runtime_records.append({
                            'trial': t,
                            'algorithm': 'MVRSM',
                            'sep_mode': args.sep,
                            'success': True,
                            'duration_sec': duration
                        })
                    except Exception as e:
                        duration = time.time() - start_time
                        runtime_records.append({
                            'trial': t,
                            'algorithm': 'MVRSM',
                            'sep_mode': args.sep,
                            'success': False,
                            'duration_sec': duration,
                            'error': str(e)
                        })
                        print(f"Error running MVRSM: {e}")

                if args.run_tpe:
                    start_time = time.time()
                    try:
                        print("Running TPE......")
                        tpe_x, tpe_fx = run_tpe(f=f, max_evals=args.max_iters, rand_evals=args.n_init, seed=trial_seed, args=args)
                        print(f"TPE completed with {len(tpe_fx)} iterations")
                        duration = time.time() - start_time
                        runtime_records.append({
                            'trial': t,
                            'algorithm': 'TPE',
                            'sep_mode': args.sep,
                            'success': True,
                            'duration_sec': duration
                        })
                    except Exception as e:
                        duration = time.time() - start_time
                        runtime_records.append({
                            'trial': t,
                            'algorithm': 'TPE',
                            'sep_mode': args.sep,
                            'success': False,
                            'duration_sec': duration,
                            'error': str(e)
                        })
                        print(f"Error running TPE: {e}")

                if args.run_rs:
                    start_time = time.time()
                    try:
                        print("Running Random Search......")
                        rs_x, rs_fx = run_random_search(f=f, max_evals=args.max_iters, seed=trial_seed, args=args)
                        print(f"Random Search completed with {len(rs_fx)} iterations")
                        duration = time.time() - start_time
                        runtime_records.append({
                            'trial': t,
                            'algorithm': 'RandomSearch',
                            'sep_mode': args.sep,
                            'success': True,
                            'duration_sec': duration
                        })
                    except Exception as e:
                        duration = time.time() - start_time
                        runtime_records.append({
                            'trial': t,
                            'algorithm': 'RandomSearch',
                            'sep_mode': args.sep,
                            'success': False,
                            'duration_sec': duration,
                            'error': str(e)
                        })
                        print(f"Error running Random Search: {e}")

                if args.run_gpyopt:
                    start_time = time.time()
                    try:
                        print("Running GPyOpt......")
                        gpyopt_x, gpyopt_fx = run_gpyopt(f=f, max_evals=args.max_iters, n_init=args.n_init,
                                                seed=trial_seed)
                        print(f"GPyOpt completed with {len(gpyopt_fx)} iterations")
                        duration = time.time() - start_time
                        runtime_records.append({
                            'trial': t,
                            'algorithm': 'GPyOpt',
                            'sep_mode': args.sep,
                            'success': True,
                            'duration_sec': duration
                        })
                    except Exception as e:
                        duration = time.time() - start_time
                        runtime_records.append({
                            'trial': t,
                            'algorithm': 'GPyOpt',
                            'sep_mode': args.sep,
                            'success': False,
                            'duration_sec': duration,
                            'error': str(e)
                        })
                        print(f"Error running GPyOpt: {e}")

                # Process result data, convert lists to numpy arrays
                def process_fx(fx_list, max_iters):
                    # Handle numpy arrays or lists
                    if isinstance(fx_list, np.ndarray):
                        fx_array = fx_list.flatten()
                    else:
                        fx_array = np.array(fx_list).flatten()
                    
                    if len(fx_array) == 0:
                        # If no results, return NaN array
                        return np.full((max_iters, 1), np.nan)
                    elif len(fx_array) < max_iters:
                        # If results are fewer than max iterations, pad with NaN
                        result = np.full((max_iters, 1), np.nan)
                        result[:len(fx_array), 0] = fx_array
                        return result
                    else:
                        # If results are sufficient, truncate to max iterations
                        return fx_array[:max_iters].reshape(-1, 1)
                
                def process_x(x_list, max_iters, dim, problem_type='mixed'):
                    """
                    Process x data, support mixed types (categorical variables as strings, continuous variables as numbers)
                    
                    Args:
                        x_list: Input data list or numpy array
                        max_iters: Maximum number of iterations
                        dim: Problem dimension
                        problem_type: Problem type, 'mixed' indicates mixed type
                    """
                    # Handle numpy arrays or lists
                    if isinstance(x_list, np.ndarray):
                        x_array = x_list
                    else:
                        x_array = np.array(x_list)
                    
                    if len(x_array) == 0:
                        # If no results, return NaN array
                        return np.full((max_iters, dim), np.nan)
                    
                    # Check if first element contains strings (categorical variables)
                    has_strings = False
                    if len(x_array) > 0 and len(x_array[0]) > 0:
                        has_strings = any(isinstance(val, str) for val in x_array[0])
                    
                    if has_strings:
                        # Mixed type: use object array
                        result = np.empty((max_iters, dim), dtype=object)
                        result.fill('n.a.')  # Fill empty values with 'n.a.'
                        
                        if len(x_array) < max_iters:
                            # If results are fewer than max iterations, pad with 'n.a.'
                            for i in range(len(x_array)):
                                for j in range(len(x_array[i])):
                                    result[i, j] = x_array[i][j]
                        else:
                            # If results are sufficient, truncate to max iterations
                            for i in range(max_iters):
                                for j in range(len(x_array[i])):
                                    result[i, j] = x_array[i][j]
                    else:
                        # Pure numeric type: use numpy array
                        # Ensure x_array is 2D
                        if x_array.ndim == 1:
                            x_array = x_array.reshape(-1, dim)
                        
                        if len(x_array) < max_iters:
                            # If results are fewer than max iterations, pad with NaN
                            result = np.full((max_iters, dim), np.nan)
                            result[:len(x_array), :] = x_array
                        else:
                            # If results are sufficient, truncate to max iterations
                            result = x_array[:max_iters]
                    
                    return result
            
                # Get problem dimension
                problem_dim = f.dim
                
                # Convert all fx arrays (order should match legend: C1G8, C3G6, C5G4, C7G2, ...)

                smk_fx_c5g4_processed = process_fx(smk_fx_c5g4, args.max_iters)
                cas_fx_processed = process_fx(cas_fx, args.max_iters)
                cocabo_fx_processed = process_fx(cocabo_fx, args.max_iters)
                
                mvrsm_fx_processed = process_fx(mvrsm_fx, args.max_iters)
                tpe_fx_processed = process_fx(tpe_fx, args.max_iters)
                rs_fx_processed = process_fx(rs_fx, args.max_iters)
                gpyopt_fx_processed = process_fx(gpyopt_fx, args.max_iters)
                
                # Convert all x arrays (order matches above)
                smk_x_c5g4_processed = process_x(smk_x_c5g4, args.max_iters, problem_dim)
                cas_x_processed = process_x(cas_x, args.max_iters, problem_dim)
                cocabo_x_processed = process_x(cocabo_x, args.max_iters, problem_dim)
                
                mvrsm_x_processed = process_x(mvrsm_x, args.max_iters, problem_dim)
                tpe_x_processed = process_x(tpe_x, args.max_iters, problem_dim)
                rs_x_processed = process_x(rs_x, args.max_iters, problem_dim)
                gpyopt_x_processed = process_x(gpyopt_x, args.max_iters, problem_dim)
                
                # Save results - each algorithm saved as a tuple of (x, fx)
                if not args.no_save:
                    new_results = [
                        (smk_x_c5g4_processed, smk_fx_c5g4_processed),  # 1: C5G4
                        (cas_x_processed, cas_fx_processed),            # 2: CASMOPOLITAN
                        (cocabo_x_processed, cocabo_fx_processed),      # 3: COCABO
                        (mvrsm_x_processed, mvrsm_fx_processed),        # 4: MVRSM
                        (tpe_x_processed, tpe_fx_processed),            # 5: TPE
                        (gpyopt_x_processed, gpyopt_fx_processed),      # 6: GPyOpt
                        (rs_x_processed, rs_fx_processed)               # 7: Random Search 
                    ]

                    # If old file exists, merge with old results
                    if os.path.exists(filename):
                        try:
                            with open(filename, 'rb') as f_old:
                                old_results = pickle.load(f_old)
                            
                            merged = []
                            for idx in range(11):
                                new_x, new_fx = new_results[idx]
                                old_x, old_fx = None, None

                                try:
                                    # If old file length is 11, match directly by index
                                    if len(old_results) >= 11:
                                        old_x, old_fx = old_results[idx]
                                    # If old file length is 10 (possibly no GPyOpt)
                                    elif len(old_results) == 10:
                                        if idx < 10:
                                            old_x, old_fx = old_results[idx]
                                        # idx == 9 (GPyOpt) or idx == 10 (RS) may have no old data
                                        elif idx == 10:
                                            # RS position (new index 10) is index 9 in old file
                                            old_x, old_fx = old_results[9]
                                    # If old file length is 9 (possibly no EDBO or GPyOpt)
                                    elif len(old_results) == 9:
                                        if idx < 9:
                                            old_x, old_fx = old_results[idx]
                                        # idx == 9 (GPyOpt) or idx == 8 (EDBO) may have no old data
                                        elif idx == 10:
                                            # RS position (new index 10) is index 8 in old file
                                            old_x, old_fx = old_results[8]
                                    # Handle other mapping cases
                                    elif len(old_results) == 6:
                                        # Old order: [C5G4, CAS, COCABO, MVRSM, TPE, RS]
                                        mapping = {2:0, 4:1, 5:2, 6:3, 7:4, 9:5}
                                        old_idx = mapping.get(idx, None)
                                        if old_idx is not None and old_idx < len(old_results):
                                            old_x, old_fx = old_results[old_idx]
                                except Exception:
                                    old_x, old_fx = None, None

                                # Determine whether to use old data
                                use_old = False
                                if old_x is not None and old_fx is not None:
                                    try:
                                        new_all_nan = False
                                        if hasattr(new_fx, 'size') and new_fx.size > 0:
                                            new_all_nan = bool(np.isnan(new_fx).all())
                                        if (new_fx is None) or (hasattr(new_fx, 'size') and new_fx.size == 0) or new_all_nan:
                                            use_old = True
                                    except Exception:
                                        pass

                                if use_old:
                                    merged.append((old_x, old_fx))
                                else:
                                    merged.append((new_x, new_fx))

                            results_to_save = merged
                            print(f"Merging results into {filename} (order: C1G8, C3G6, C5G4, C7G2, CAS, COCABO, MVRSM, TPE, EDBO, GPyOpt, RS)...")
                        except Exception as e:
                            print(f"Failed to merge with existing file, saving new results only. Reason: {e}")
                            results_to_save = new_results
                    else:
                        results_to_save = new_results

                    with open(filename, 'wb') as file:
                        pickle.dump(results_to_save, file)
                    print("Saving optimization path analysis results (x and fx) to %s..." % filename)

                if args.seed is not None:
                    args.seed += 1

                print("Optimization path analysis completed!")
                if runtime_records:
                    runtime_df = pd.DataFrame(runtime_records)
                    runtime_csv_path = os.path.join(save_dir, 'runtime_summary.csv')
                    runtime_df.to_csv(runtime_csv_path, index=False)
                    print(f"Saved runtime summary to {runtime_csv_path}")

                # Clear memory after each problem
                clear_memory()


if __name__ == "__main__":
    try:
        run_optimization_path_analysis()
    finally:
        # Final memory cleanup when script ends
        clear_memory()
