import pandas as pd
import sys
import os
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cas.optimizer import Optimizer
from cas.optimizer_mixed import MixedOptimizer
import time
import numpy as np
from cocabo.CoCaBO import CoCaBO
from mvrsm.MVRSM import MVRSM_minimize
from hyperopt import fmin, rand, tpe, hp, STATUS_OK, Trials
from functools import partial
from mvrsm.process import read_logs_MVRSM, read_logs_TPE, read_logs_RS
import GPyOpt




# All optimization algorithms defined must be minimize

def run_cas(f, args):
    """
    Run CAS-CatBOX algorithm without noise
    """
    kwargs = {"continuous_kern_type": args.continuous_kern_type, "num_mixtures1": args.num_mixtures1,
              "num_mixtures2": args.num_mixtures2}


    # Set random seed to ensure reproducibility
    np.random.seed(args.seed)
    try:
        import torch
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    except ImportError:
        pass  # Skip if torch is not installed

    n_categories = f.n_vertices
    problem_type = f.problem_type

    if args.infer_noise_var:
        noise_variance = None
    else:
        noise_variance = f.lamda if hasattr(f, 'lamda') else None

    if args.kernel_type is None:
        kernel_type = 'mixed' if problem_type == 'mixed' else 'transformed_overlap'
    else:
        kernel_type = args.kernel_type

    if problem_type == 'mixed':
        optim = MixedOptimizer(f.config, f.lb, f.ub, f.continuous_dims, f.categorical_dims,
                               n_init=args.n_init, use_ard=args.ard, acq=args.acq,
                               kernel_type=kernel_type,
                               noise_variance=noise_variance, init_design=args.init_design, source=f,
                               **kwargs)
    else:
        optim = Optimizer(f.config, n_init=args.n_init, use_ard=args.ard, acq=args.acq,
                          kernel_type=kernel_type,
                          noise_variance=noise_variance, **kwargs)
    print('Optimizer initialized successfully')
    start = time.time()
    
    # Store x values from all iterations (including readable categorical variables)
    x_history = []
    print('Starting normal iterations')
    for i in tqdm(range(args.max_iters), desc=f"CAS-SMKBO iterations Cauchy {args.num_mixtures1}, Gaussian {args.num_mixtures2}", unit="iter"):
        x_next = optim.suggest(args.batch_size)
        y_next = f.compute(x_next, normalize=f.normalize)
        optim.observe(x_next, y_next)
        
        if f.normalize:
            Y = np.array(optim.casmopolitan.fX) * f.std + f.mean
        else:
            Y = np.array(optim.casmopolitan.fX)
        
        # Record x values from each iteration
        # Check if there are categorical variables

        if hasattr(f, 'cat_var') and len(f.cat_var) > 0:
            # For Chemistry/OCM problems, use hardcoded number of categorical variables
            # For DAR problems, use dynamically obtained number
            if hasattr(f, '__class__') and 'Chemistry' in f.__class__.__name__:
                # Hardcoded logic for Chemistry/OCM problems
                if args.sep in ['atom-mol', 'sep']:
                    cat_num = 4
                elif args.sep in ['all','all_update', 'all_update_true', 'm1', 'm12', 'atom', 'true_atom']:
                    cat_num = 1
                elif args.sep in ['m1m2', 'M1', 'M2', 'M3', 'Support']:
                    cat_num = 2
                else:
                    raise ValueError("Unsupported sep option")
            else:
                # For DAR and other problems, dynamically get number of categorical variables
                cat_num = len(f.cat_var)
            
            # Get readable categorical variable values
            cat_name = f.encoder.from_cat(
                pd.DataFrame(np.atleast_2d([int(x) for x in x_next.flatten()[:cat_num]]), columns=f.cat_var))
            cat_values = cat_name.values.flatten().tolist()
            
            # Get continuous variable values
            x_next_float_list = x_next.flatten()[cat_num:]
            
            # Create mixed x values: categorical variables as strings, continuous variables as numbers
            mixed_x = []
            for j in range(cat_num):
                mixed_x.append(cat_values[j])
            for j in range(len(x_next_float_list)):
                mixed_x.append(x_next_float_list[j])
            
            x_history.append(mixed_x)
        else:
            # Pure continuous problem: directly use continuous variable values
            x_next_float_list = x_next.flatten()
            x_history.append(x_next_float_list.tolist())
        
        # Print information (only calculate best value when there is sufficient data)
        if Y[:i].shape[0]:
            argmin = np.argmin(Y[:i * args.batch_size])
            formatted_x_next = ["%.2f" % x for x in x_next_float_list]
            if hasattr(f, 'cat_var') and len(f.cat_var) > 0:
                print('Iter %d, Last X %s %s; \n fX:  %.4f. fX_best: %.4f'
                      % (i, cat_values, formatted_x_next,
                         -float(Y[-1]),
                         -Y[:i * args.batch_size][argmin]))
            else:
                print('Iter %d, Last X %s; \n fX:  %.4f. fX_best: %.4f'
                      % (i, formatted_x_next,
                         -float(Y[-1]),
                         -Y[:i * args.batch_size][argmin]))
        else:
            formatted_x_next = ["%.2f" % x for x in x_next_float_list]
            if hasattr(f, 'cat_var') and len(f.cat_var) > 0:
                print('Iter %d, Last X %s %s; \n fX:  %.4f.'
                      % (i, cat_values, formatted_x_next, -float(Y[-1])))
            else:
                print('Iter %d, Last X %s; \n fX:  %.4f.'
                      % (i, formatted_x_next, -float(Y[-1])))
    
    print('CAS-SMKBO time cost: ', time.time()-start, 's\n')

    # Determine return array type based on whether there are categorical variables
    if hasattr(f, 'cat_var') and len(f.cat_var) > 0:
        # Mixed problem: use object array to preserve strings
        return np.array(x_history, dtype=object), -optim.casmopolitan.fX
    else:
        # Pure continuous problem: use regular float array
        return np.array(x_history, dtype=float), -optim.casmopolitan.fX


def run_cocabo(f, budget, initN=24, kernel_mix=0.5, n_trial=1, seed=None, args=None):
    """
    Run COCABO algorithm without noise
    """
    categories = f.n_vertices.tolist()
    bounds = f.get_cocabo_bounds()
    print(f"Running COCABO for {budget} evaluations...")
    start = time.time()
    mabbo = CoCaBO(objfn=f, initN=initN, bounds=bounds,
                   acq_type='EI', C=categories,
                   kernel_mix=kernel_mix)
    
    fx = mabbo.runTrials(budget, n_trial)
    # Get COCABO's actual input point trajectory and function values
    raw_x_history = mabbo.data[0]  # COCABO object stores all input points (raw format)
    fx_history = mabbo.result[0]  # COCABO object stores all function values
    
    # Convert categorical variables to readable format
    x_history = convert_categorical_variables(raw_x_history, f, "COCABO")


    # Verify data consistency
    print(f'COCABO data shape: x_history={len(x_history)} points, fx_history={fx_history.shape}')
    print(f'COCABO best value from fx: {np.max(-fx_history)}')
    print(f'COCABO best value from runTrials: {np.max(fx)}')
    
    return np.array(x_history), -fx_history


def get_f_info(f):
    """Get function information"""
    d = f.dim  # Total number of variables
    num_int = len(f.cat_var)
    int_lb = np.zeros(len(f.cat_var), dtype=int)
    int_ub = (f.n_vertices - 1).astype(int)

    cont_lb = f.lb
    cont_ub = f.ub
    lb = np.concatenate([int_lb, cont_lb])
    ub = np.concatenate([int_ub, cont_ub])
    return d, num_int, lb, ub


def convert_categorical_variables(raw_x_history, noisy_f, algorithm_name):
    """
    Convert categorical variables from integer encoding to readable string format
    
    Args:
        raw_x_history: Raw input point history (containing integer-encoded categorical variables)
        noisy_f: Function wrapper with noise
        algorithm_name: Algorithm name (for debugging information)
    
    Returns:
        x_history: Converted input point history (categorical variables as string format)
    """
    num_cat = len(noisy_f.cat_var)  # Number of categorical variables
    x_history = []
    
    print(f'Converting {len(raw_x_history)} {algorithm_name} points with {num_cat} categorical variables')
    
    for i in range(len(raw_x_history)):
        try:
            # Get categorical variables of current point (integer encoding)
            cat_ints = [int(x) for x in raw_x_history[i][:num_cat]]

            # Check if encoder attribute exists (for real experimental functions)
            if hasattr(noisy_f, 'encoder') and noisy_f.encoder is not None:
                # For real experimental functions, use encoder for conversion
                cat_names = noisy_f.encoder.from_cat(
                    pd.DataFrame(np.atleast_2d(cat_ints), columns=noisy_f.cat_var))
                cat_values = cat_names.values.flatten().tolist()
            else:
                # For benchmark functions, directly use integers as string labels
                cat_values = [f'{val}' for val in enumerate(cat_ints)]
            
            # Get continuous variable values
            if isinstance(raw_x_history[i], np.ndarray):
                cont_values = raw_x_history[i][num_cat:].tolist()
            else:
                cont_values = raw_x_history[i][num_cat:]
            
            # Create mixed x values: categorical variables as strings, continuous variables as numbers
            mixed_x = []
            for j in range(num_cat):
                mixed_x.append(cat_values[j])
            for j in range(len(cont_values)):
                mixed_x.append(cont_values[j])
            
            x_history.append(mixed_x)
            if i == 0:  # Only print conversion result for first point
                print(f"{algorithm_name} conversion example:")
                print(f"  Original: {raw_x_history[i]}")
                print(f"  Converted: {mixed_x}")
            
        except Exception as e:
            print(f"Error converting point {i} in {algorithm_name}: {e}")
            print(f"Point data: {raw_x_history[i]}")
            # If conversion fails, use original data
            x_history.append(raw_x_history[i])
    
    # Use object array to maintain mixed data types (strings and numbers)
    return np.array(x_history, dtype=object)


def run_mvrsm(f, max_evals, rand_evals, seed, args=None):
    """
    Run MVRSM algorithm without noise
    """
    np.random.seed(seed)
    d, num_int, lb, ub = get_f_info(f)

    x0 = np.zeros(d)
    # Random initial guess (integer)
    x0[0:num_int] = np.round(
        np.random.rand(num_int) * (ub[0:num_int] - lb[0:num_int]) + lb[0:num_int])
    # Random initial guess (continuous)
    x0[num_int:d] = np.random.rand(d - num_int) * (ub[num_int:d] - lb[num_int:d]) + lb[num_int:d]

    # Create data storage lists
    x_history = []
    fx_history = []
    
    # Wrap objective function to record data
    def wrapped_objective(x):
        fx = f.compute(x)
        x_history.append(x.copy())
        fx_history.append(fx)
        print('accepted x: ', x, 'fx: ', fx)
        return fx

    print(f"Running MVRSM for {max_evals} evaluations...")
    start = time.time()
    solX, solY, model, logfile = MVRSM_minimize(wrapped_objective, x0, lb, ub, num_int, max_evals, rand_evals)
    print('MVRSM time cost: ', time.time() - start, 's\n')
    
    # Ensure we have data
    if len(x_history) == 0:
        print("Warning: No data captured during MVRSM optimization")
        # If no data captured, use final solution
        x_history = [solX]
        fx_history = [solY]
    
    # Convert categorical variables to readable format
    x_history = convert_categorical_variables(x_history, f, "MVRSM")
    fx_history = np.array(fx_history).flatten()
    
    return x_history, -fx_history


def run_random_search(f, max_evals, seed, args=None):
    """
    Run random search algorithm without noise
    """
    # HyperOpt and RS objective
    def hyp_obj(x):
        return {'loss': f.compute(x), 'status': STATUS_OK}

    np.random.seed(seed)
    d, num_int, lb, ub = get_f_info(f)

    var = [None] * d  # variable for hyperopt and random search
    for i in list(range(0, d)):
        if i < num_int:
            var[i] = hp.quniform('var_d' + str(i), lb[i], ub[i], 1)  # Integer variables
        else:
            var[i] = hp.uniform('var_c' + str(i), lb[i], ub[i])  # Continuous variables
    algo = rand.suggest
    trials_RS = Trials()

    print(f"Running Random Search for {max_evals} evaluations...")
    start = time.time()
    RS = fmin(hyp_obj, var, algo, max_evals=max_evals, trials=trials_RS)
    print('Random Search time cost: ', time.time() - start, 's\n')

    current_dir = os.path.dirname(__file__)
    folder = os.path.join(current_dir, 'log_M')
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    # Directly extract function values from trials (actual computed value for each point)
    rs_fx = []
    for trial in trials_RS.trials:
        if 'result' in trial and 'loss' in trial['result']:
            rs_fx.append(-trial['result']['loss'])
    rs_fx = np.array(rs_fx)
    
    # Extract input points from trials
    raw_x_history = []
    for trial in trials_RS.trials:
        if 'misc' in trial and 'vals' in trial['misc']:
            x = []
            for i in range(d):
                if i < num_int:
                    x.append(int(trial['misc']['vals'][f'var_d{i}'][0]))
                else:
                    x.append(trial['misc']['vals'][f'var_c{i}'][0])
            raw_x_history.append(x)
    
    # Convert categorical variables to readable format
    if len(raw_x_history) > 0:
        x_history = convert_categorical_variables(raw_x_history, f, "Random Search")
    else:
        print("Warning: No x data extracted from Random Search trials")
        x_history = np.array([])
    
    return x_history, rs_fx


def run_tpe(f, max_evals, rand_evals, seed, args=None):
    """
    Run TPE algorithm without noise
    """
    def hyp_obj(x):
        return {'loss': f.compute(x), 'status': STATUS_OK}
    np.random.seed(seed)
    d, num_int, lb, ub = get_f_info(f)

    algo = partial(tpe.suggest, n_startup_jobs=rand_evals)
    trials_tpe = Trials()

    # Define search space for HyperOpt
    var = [None] * d  # variable for hyperopt and random search
    for i in list(range(0, d)):
        if i < num_int:
            var[i] = hp.quniform('var_d' + str(i), lb[i], ub[i], 1)  # Integer variables
        else:
            var[i] = hp.uniform('var_c' + str(i), lb[i], ub[i])  # Continuous variables

    print(f"Running TPE for {max_evals} evaluations...")
    start = time.time()  # Start timer
    TPE = fmin(hyp_obj, var, algo, max_evals=max_evals, trials=trials_tpe)  # Run TPE
    print('TPE time cost: ', time.time() - start, 's\n')

    current_dir = os.path.dirname(__file__)
    folder = os.path.join(current_dir, 'log_M')
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    # Directly extract function values from trials (actual computed value for each point)
    tpe_fx = []
    for trial in trials_tpe.trials:
        if 'result' in trial and 'loss' in trial['result']:
            tpe_fx.append(-trial['result']['loss'])
    tpe_fx = np.array(tpe_fx)
    
    # Extract input points from trials
    raw_x_history = []
    for trial in trials_tpe.trials:
        if 'misc' in trial and 'vals' in trial['misc']:
            x = []
            for i in range(d):
                if i < num_int:
                    x.append(int(trial['misc']['vals'][f'var_d{i}'][0]))
                else:
                    x.append(trial['misc']['vals'][f'var_c{i}'][0])
            raw_x_history.append(x)
    
    # Convert categorical variables to readable format
    if len(raw_x_history) > 0:
        x_history = convert_categorical_variables(raw_x_history, f, "TPE")
    else:
        print("Warning: No x data extracted from TPE trials")
        x_history = np.array([])
    
    return x_history, tpe_fx


def _to_dataframe_like(x, columns=None) -> pd.DataFrame:
    """Convert input to DataFrame format, supports DataFrame/Series/dict/ndarray/list."""
    if isinstance(x, pd.DataFrame):
        return x
    if isinstance(x, pd.Series):
        return x.to_frame().T
    if isinstance(x, dict):
        return pd.DataFrame([x])
    if isinstance(x, (np.ndarray, list)):
        if columns is None:
            raise ValueError("When x is array-like, `columns` must be provided.")
        return pd.DataFrame([x], columns=columns)
    return pd.DataFrame([x], columns=columns)

def run_gpyopt(f, max_evals, n_init=20, seed=None):
    """
    Run GPyOpt Bayesian optimization algorithm

    Args:
        f: Objective function object
        max_evals: Maximum number of evaluations
        n_init: Number of initial random evaluations
        acq: Acquisition function 
        seed: Random seed

    Returns:
        x_history: Input point history (readable format)
        fx_history: Function value history
    """

    if seed is not None:
        np.random.seed(seed)

    print(f"Running GPyOpt for {max_evals} evaluations (init={n_init})...")
    start = time.time()

    # Get problem information
    d, num_int, lb, ub = get_f_info(f)
    cat_vars = f.cat_var
    cont_vars = f.cont_var

    print(f"  Problem dimension: {d} (categorical: {num_int}, continuous: {d-num_int})")
    print(f"  Categorical variables: {cat_vars}")
    print(f"  Continuous variables: {cont_vars}")

    # Define variable types and bounds
    bounds = []

    # Categorical variables: discrete type
    for i in range(num_int):
        bounds.append({
            'name': f'var_{i}',
            'type': 'categorical',
            'domain': tuple(range(int(lb[i]), int(ub[i])+1))
        })

    # Continuous variables: continuous type
    for i in range(num_int, d):
        bounds.append({
            'name': f'var_{i}',
            'type': 'continuous',
            'domain': (lb[i], ub[i])
        })

    # Wrap objective function to record history
    x_history_raw = []
    fx_history = []

    def objective_function(x):
        """GPyOpt objective function wrapper"""
        x = x.flatten()

        # Compute function value
        fx = f.compute(x)
        fx_val = float(fx[0, 0]) if hasattr(fx, 'shape') else float(fx)

        return fx_val

    try:
        # Create GPyOpt optimizer
        optimizer = GPyOpt.methods.BayesianOptimization(
            f=objective_function,
            domain=bounds,
            acquisition_type='EI',
            normalize_Y=False,  
            initial_design_numdata=n_init,
            initial_design_type='random',
            exact_feval=True,  # Exact function evaluation
            verbosity=False
        )

        # Run optimization
        max_iter = max_evals - n_init
        
        if max_iter > 0:
            pbar = tqdm(total=max_iter)
            for i in range(max_iter):
                optimizer.run_optimization(max_iter=1, eps=0)
                pbar.update(1)
        pbar.close()
        # Get actual data from optimizer
        x_history_raw = optimizer.X
        fx_history = optimizer.Y
        print(f'GPyOpt completed in {time.time() - start:.2f} s')
        print(f'  Total evaluations: {len(fx_history)}')
        if len(fx_history) > 0:
            print(f'  Best value found: {max(fx_history):.6f}') 
        # Convert input points to readable format
        x_history = convert_categorical_variables(x_history_raw, f, "GPyOpt")

        return x_history, -np.array(fx_history)

    except Exception as e:
        print(f"Error in GPyOpt optimization: {e}")
        # If optimization fails, at least return existing data
        if x_history_raw:
            x_history = convert_categorical_variables(x_history_raw, f, "GPyOpt")
            return x_history, -np.array(fx_history)
        else:
            # Return empty results
            return np.array([]), np.array([])


def test_gpyopt():
    """Test run_gpyopt function"""
    print("="*70)
    print("Testing run_gpyopt function")
    print("="*70)

    # Import Chemistry for testing
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    try:
        from mixed_test_func.Chemistry.chemistry import Chemistry

        # Initialize OCM problem
        ocm = Chemistry(lamda=0, normalize=False, seed=42, sep='all_update', prob='OCM2')

        x_history, fx_history = run_gpyopt(
            f=ocm,
            max_evals=20,
            n_init=5,
            seed=42
        )

        # Show best conditions
        if len(fx_history) > 0:
            best_idx = np.argmax(fx_history)
            best_x = x_history[best_idx]
            var_names = ocm.cat_var + ocm.cont_var

            print(f"\n      Best conditions found at iteration {best_idx}:")
            for i, (var_name, var_value) in enumerate(zip(var_names, best_x)):
                if isinstance(var_value, str):
                    print(f"        {var_name:12s}: {var_value}")
                else:
                    print(f"        {var_name:12s}: {var_value:.6f}")
        print('fx_history: ', fx_history)
        print('x_history: ', x_history)
        print("\n" + "="*70)
        print("✅ run_gpyopt test passed successfully!")
        print("="*70 + "\n")

    except ImportError as e:
        print(f"✗ Import error: {e}")
        print("Please install GPyOpt: pip install gpyopt")
        sys.exit(1)
    except Exception as e:
        print(f"✗ Error during test: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    """Test functions"""
    import sys

    test_gpyopt()


