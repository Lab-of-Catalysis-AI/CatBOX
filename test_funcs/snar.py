"""
Snar Benchmark - Nucleophilic Aromatic Substitution Reaction
A continuous optimization problem from Summit benchmarks
"""
import numpy as np
from .base import TestFunction

try:
    from summit.benchmarks import SnarBenchmark
    from summit.utils.dataset import DataSet
    SUMMIT_AVAILABLE = True
except ImportError:
    SUMMIT_AVAILABLE = False
    print("Warning: Summit not available. Snar benchmark cannot be used.")


class Snar(TestFunction):
    """
    Snar Benchmark - Nucleophilic Aromatic Substitution Reaction
    
    This is a pure continuous optimization problem with 4 input variables:
    - tau: residence time (0.5 to 2.0 min)
    - equiv_pldn: equivalents of pyrrolidine (1.0 to 5.0)
    - conc_dfnb: concentration of DFNB (0.1 to 0.5 M)
    - temperature: reaction temperature (30 to 120 °C)
    
    Objectives (converted to single objective for minimization):
    - STY (space-time yield): to maximize
    - E-factor: to minimize
    
    Single objective: -STY/1e4 + E-factor/100 (minimize)
    """
    
    problem_type = 'mixed'  # Use 'mixed' to trigger MixedOptimizer
    
    def __init__(self, normalize=False, lamda=1e-6, seed=None):
        """
        Initialize Snar benchmark
        
        Args:
            normalize: Whether to normalize the objective values
            lamda: Noise level (for compatibility, not used in Snar)
            seed: Random seed (for compatibility)
        """
        if not SUMMIT_AVAILABLE:
            raise ImportError("Summit package is required for Snar benchmark. "
                            "Install it with: pip install summit")
        
        super(Snar, self).__init__(normalize=normalize)
        
        # Initialize Summit Snar benchmark
        self._exp = SnarBenchmark()
        
        # Get input variable information
        self.input_vars = [v.name for v in self._exp.domain.input_variables]
        d_cont = len(self.input_vars)  # Should be 4 continuous variables
        
        # Extract bounds for continuous variables
        self.lb = np.array([float(v.lower_bound) for v in self._exp.domain.input_variables])
        self.ub = np.array([float(v.upper_bound) for v in self._exp.domain.input_variables])
        
        # Problem dimensions
        # Add 1 dummy categorical variable (2 categories: 0, 1) that doesn't affect the objective
        # This allows MixedOptimizer to work properly with its MixtureKernel
        self.dim = d_cont + 1  # 4 continuous + 1 dummy categorical
        self.categorical_dims = [0]  # First dimension is categorical (dummy)
        self.continuous_dims = list(range(1, self.dim))  # Remaining dimensions are continuous
        
        # Config: 1 categorical variable with 2 categories
        self.config = np.array([2], dtype=int)
        self.n_vertices = [2]
        
        # Other attributes
        self.lamda = lamda
        self.seed = seed
        
        # Normalization
        if normalize:
            self.mean, self.std = self.sample_normalize(size=50)
        else:
            self.mean, self.std = None, None
    
    def _evaluate_continuous(self, x_continuous):
        """
        Evaluate the Snar benchmark at continuous points
        
        Args:
            x_continuous: numpy array of shape (n, d) with continuous values
        
        Returns:
            objectives: numpy array of shape (n,) with scalar objective values
        """
        if x_continuous.ndim == 1:
            x_continuous = x_continuous.reshape(1, -1)
        
        results = []
        
        for i in range(x_continuous.shape[0]):
            x_row = x_continuous[i].copy()
            
            # Clip to bounds
            x_row = np.clip(x_row, self.lb, self.ub)
            
            # Safeguard: ensure tau (first variable) is not too small
            if x_row[0] < 1e-6:
                x_row[0] = 1e-6
            
            try:
                # Create Summit DataSet
                values = {(name, "DATA"): float(val) 
                         for name, val in zip(self.input_vars, x_row)}
                ds = DataSet([values], columns=self.input_vars)
                
                # Run experiment
                res = self._exp.run_experiments(ds)
                
                # Extract objectives
                sty = float(res[("sty", "DATA")].iloc[-1])
                e_factor = float(res[("e_factor", "DATA")].iloc[-1])
                
                # Scalar objective: minimize (-STY/1e4 + E-factor/100)
                scalar = -sty / 1e4 + e_factor / 100.0
                results.append(scalar)
                
            except Exception as e:
                # If evaluation fails, return a large penalty
                print(f"Warning: evaluation failed for x={x_row}, error={e}")
                results.append(1e6)
        
        return np.array(results)
    
    def compute(self, x, normalize=False):
        """
        Compute objective value for mixed input (categorical + continuous)
        
        Args:
            x: numpy array where first column is dummy categorical (ignored),
               remaining columns are continuous values in [lb, ub]
            normalize: whether to normalize the output
        
        Returns:
            objective values (as numpy array for compatibility with optimizers)
        """
        if normalize is None:
            normalize = self.normalize
        
        x = np.asarray(x)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        # Extract continuous part (ignore first column which is dummy categorical)
        # x shape: (batch, 5) where x[:, 0] is dummy categorical, x[:, 1:5] are continuous
        x_continuous = x[:, 1:]  # Skip the dummy categorical variable
        
        # Evaluate
        res = self._evaluate_continuous(x_continuous)
        
        # Normalize if requested
        if normalize and self.mean is not None and self.std is not None:
            res = (res - self.mean) / self.std
        
        return res.reshape(-1, 1)
    
    def sample_normalize(self, size=None):
        """
        Sample random points to estimate mean and std for normalization
        
        Args:
            size: number of samples
        
        Returns:
            mean, std: normalization parameters
        """
        if size is None:
            size = 2 * self.dim + 1
        
        y = []
        for i in range(size):
            # Sample: 1 dummy categorical + 4 continuous values
            x_cat = np.random.randint(0, 2, size=1)  # Dummy categorical (0 or 1)
            x_cont = np.random.uniform(self.lb, self.ub)  # 4 continuous values
            x = np.concatenate([x_cat, x_cont])
            y.append(self.compute(x, normalize=False)[0, 0])
        
        y = np.array(y)
        return np.mean(y), np.std(y)


# For testing
if __name__ == "__main__":
    print("Testing Snar benchmark...")
    
    # Create instance
    snar = Snar(normalize=False)
    
    print(f"Problem type: {snar.problem_type}")
    print(f"Problem dimension: {snar.dim} (1 dummy categorical + 4 continuous)")
    print(f"Input variables: {snar.input_vars}")
    print(f"Bounds (continuous part): lb={snar.lb}, ub={snar.ub}")
    print(f"Config: {snar.config} (1 categorical with 2 categories)")
    print(f"Continuous dims: {snar.continuous_dims}")
    print(f"Categorical dims: {snar.categorical_dims}")
    
    # Test with mixed input (dummy categorical + continuous)
    x_cat = np.array([0])  # Dummy categorical = 0
    x_cont = np.array([0.5,5.0,0.5,46.58115464779652])
    # x_cont = (snar.lb + snar.ub) / 2
    x_mixed = np.concatenate([x_cat, x_cont])
    print(f"\nTesting mixed input: cat={x_cat[0]}, cont={x_cont}")
    y = snar.compute(x_mixed)
    print(f"Objective value: {y[0, 0]:.6f}")
    
    # Test with different dummy categorical value (should give same result)
    x_cat2 = np.array([1])  # Dummy categorical = 1
    x_mixed2 = np.concatenate([x_cat2, x_cont])
    print(f"\nTesting mixed input: cat={x_cat2[0]}, cont={x_cont}")
    y2 = snar.compute(x_mixed2)
    print(f"Objective value: {y2[0, 0]:.6f}")
    print(f"✓ Values match (dummy cat ignored): {np.abs(y[0,0] - y2[0,0]) < 1e-10}")
    
    print("\n✓ Snar benchmark test completed!")

