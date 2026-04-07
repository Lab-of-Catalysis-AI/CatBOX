"""Base class for pure continuous optimization problems"""
import numpy as np
from test_funcs.base import TestFunction


class ContinuousFunction(TestFunction):
    """Base class for pure continuous optimization problems"""
    problem_type = 'continuous'
    
    def __init__(self, dim, lb, ub, normalize=False, lamda=1e-6, seed=None):
        """
        Initialize pure continuous optimization problem
        
        Args:
            dim: Variable dimension
            lb: Lower bounds (array or list)
            ub: Upper bounds (array or list)
            normalize: Whether to normalize
            lamda: Noise level
            seed: Random seed
        """
        super().__init__(normalize=normalize)
        self.dim = dim
        self.lb = np.array(lb)
        self.ub = np.array(ub)
        self.lamda = lamda
        self.seed = seed
        self.normalize = normalize
        
        # Attributes specific to pure continuous problems
        self.cat_var = []  # Empty list indicates no categorical variables
        self.cont_var = [f'x{i}' for i in range(dim)]  # Continuous variable names
        self.categorical_dims = np.array([])  # Empty array
        self.continuous_dims = np.arange(dim)  # All dimensions are continuous
        self.n_vertices = np.array([])  # No categorical variables
        self.config = []  # Empty configuration
        
        if seed is not None:
            np.random.seed(seed)
        
        if self.normalize:
            self.mean, self.std = self.sample_normalize()
        else:
            self.mean, self.std = None, None
    
    def compute(self, X, normalize=False, minimize=True):
        """
        Evaluate function value (subclass must override)
        
        Args:
            X: Input points, shape (n_samples, dim) or (dim,)
            normalize: Whether to normalize results
            minimize: Whether it's a minimization problem
        
        Returns:
            Function values, shape (n_samples, 1)
        """
        raise NotImplementedError("Subclass must implement compute() method")
    
    def sample_normalize(self, size=None):
        """Compute normalization parameters"""
        if size is None:
            size = 2 * self.dim + 1
        
        # Latin hypercube sampling
        from cas.localbo_utils import latin_hypercube, from_unit_cube
        x_samples = latin_hypercube(size, self.dim)
        x_samples = from_unit_cube(x_samples, self.lb, self.ub)
        
        y = []
        for i in range(size):
            y.append(self.compute(x_samples[i], normalize=False))
        y = np.array(y)
        
        return np.mean(y), np.std(y)
    
    def get_cocabo_bounds(self):
        """Get COCABO format bounds (continuous variables only)"""
        bounds = []
        for i, var_name in enumerate(self.cont_var):
            domain = (float(self.lb[i]), float(self.ub[i]))
            bounds.append({
                'name': var_name,
                'type': 'continuous',
                'domain': domain
            })
        return bounds


class Sphere(ContinuousFunction):
    """Sphere test function: f(x) = sum(x_i^2)"""
    
    def __init__(self, dim=10, normalize=False, lamda=1e-6, seed=None):
        lb = -5.12 * np.ones(dim)
        ub = 5.12 * np.ones(dim)
        super().__init__(dim, lb, ub, normalize, lamda, seed)
        self.optimum = 0.0
        self.optimum_x = np.zeros(dim)
    
    def compute(self, X, normalize=False, minimize=True):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        res = np.sum(X ** 2, axis=1, keepdims=True)
        res += self.lamda * np.random.rand(*res.shape)
        
        if normalize:
            res = (res - self.mean) / self.std
        
        return -res if not minimize else res


class Rosenbrock(ContinuousFunction):
    """Rosenbrock test function"""
    
    def __init__(self, dim=10, normalize=False, lamda=1e-6, seed=None):
        lb = -5 * np.ones(dim)
        ub = 10 * np.ones(dim)
        super().__init__(dim, lb, ub, normalize, lamda, seed)
        self.optimum = 0.0
        self.optimum_x = np.ones(dim)
    
    def compute(self, X, normalize=False, minimize=True):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        res = np.zeros((X.shape[0], 1))
        for i in range(X.shape[1] - 1):
            res += (100 * (X[:, i+1] - X[:, i]**2)**2 + (1 - X[:, i])**2).reshape(-1, 1)
        
        res += self.lamda * np.random.rand(*res.shape)
        
        if normalize:
            res = (res - self.mean) / self.std
        
        return -res if not minimize else res


class Ackley(ContinuousFunction):
    """Ackley test function"""
    
    def __init__(self, dim=10, normalize=False, lamda=1e-6, seed=None):
        lb = -32.768 * np.ones(dim)
        ub = 32.768 * np.ones(dim)
        super().__init__(dim, lb, ub, normalize, lamda, seed)
        self.optimum = 0.0
        self.optimum_x = np.zeros(dim)
    
    def compute(self, X, normalize=False, minimize=True):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        a = 20
        b = 0.2
        c = 2 * np.pi
        
        sum1 = np.sum(X ** 2, axis=1)
        sum2 = np.sum(np.cos(c * X), axis=1)
        
        term1 = -a * np.exp(-b * np.sqrt(sum1 / self.dim))
        term2 = -np.exp(sum2 / self.dim)
        
        res = (term1 + term2 + a + np.e).reshape(-1, 1)
        res += self.lamda * np.random.rand(*res.shape)
        
        if normalize:
            res = (res - self.mean) / self.std
        
        return -res if not minimize else res

