
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import sys
import os
import warnings

warnings.filterwarnings('ignore')

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from test_funcs.base import TestFunction


#=========================================================================

class CategoricalEncoder:
    def __init__(self):
        self.encoders = {}
        self.column_dtypes = {}

    def to_cat(self, df):
        encoded_df = df.copy()
        self.encoders = {}
        self.column_dtypes = {}

        for col in encoded_df.columns:
            self.column_dtypes[col] = str(encoded_df[col].dtype)
            le = LabelEncoder()
            encoded_df[col] = le.fit_transform(encoded_df[col].astype(str))
            self.encoders[col] = le
        return encoded_df

    def from_cat(self, encoded_df):
        if not self.encoders:
            raise ValueError("No encoding information. Use to_cat() first")

        decoded_df = encoded_df.copy()

        for col in decoded_df.columns:
            if col in self.encoders:
                decoded_df[col] = self.encoders[col].inverse_transform(decoded_df[col])
                if self.column_dtypes[col] == 'category':
                    decoded_df[col] = decoded_df[col].astype('category')
                elif 'int' in self.column_dtypes[col]:
                    try:
                        decoded_df[col] = decoded_df[col].astype(self.column_dtypes[col])
                    except:
                        decoded_df[col] = pd.to_numeric(decoded_df[col], errors='ignore')
                elif 'float' in self.column_dtypes[col]:
                    decoded_df[col] = decoded_df[col].astype(self.column_dtypes[col])
                elif 'bool' in self.column_dtypes[col]:
                    decoded_df[col] = decoded_df[col].astype(bool)

        return decoded_df


#=========================================================================

class Ackley_benchmark(TestFunction):
    """
    Ackley function for mixed categorical and continuous variables
    Categorical variables: discretized continuous space into 20 bins
    Continuous variables: in [-32.768, 32.768]
    """
    problem_type = 'mixed'

    def __init__(self, n_categorical=1, n_continuous=1, num_opts=20, lamda=1e-6, normalize=False, seed=None):
        super().__init__(normalize=normalize)
        self.current_dir = os.path.dirname(__file__)
        self.seed = seed
        self.normalize = normalize
        self.lamda = lamda
        self.n_categorical = n_categorical
        self.n_continuous = n_continuous
        self.num_opts = num_opts  # Number of options for categorical variables

        # Set dimension information
        self.dim = n_categorical + n_continuous
        self.categorical_dims = np.arange(n_categorical) if n_categorical > 0 else np.array([])
        self.continuous_dims = np.arange(n_categorical, self.dim) if n_continuous > 0 else np.array([])

        # Set options for categorical variables
        self.n_vertices = np.array([num_opts] * n_categorical) if n_categorical > 0 else np.array([])
        self.config = self.n_vertices

        # Set variable names
        self.cat_var = [f'x_{i}' for i in range(n_categorical)] if n_categorical > 0 else []
        self.cont_var = [f'x_{i}' for i in range(n_categorical, self.dim)] if n_continuous > 0 else []

        # Set bounds for continuous variables
        if n_continuous > 0:
            self.cont_bounds = [(-32.768, 32.768)] * n_continuous
            self.lb = np.array([b[0] for b in self.cont_bounds])
            self.ub = np.array([b[1] for b in self.cont_bounds])
        else:
            self.cont_bounds = []
            self.lb = np.array([])
            self.ub = np.array([])

        # Initialize encoder
        self.encoder = CategoricalEncoder()

        # Set encoding information for benchmark function (simulate to_cat call)
        if self.n_categorical > 0:

            virtual_cat_data = {}
            for var_name in self.cat_var:

                virtual_cat_data[var_name] = list(range(self.num_opts))

            virtual_df = pd.DataFrame(virtual_cat_data)
            self.encoder.to_cat(virtual_df)
        
        # Normalization settings
        if self.normalize:
            self.mean, self.std = self.sample_normalize()
        else:
            self.mean, self.std = None, None

    def ackley(self, vector, a=20., b=0.2, c=2. * np.pi):
        dim = len(vector)
        result = - a * np.exp( - b * np.sqrt( np.sum(vector**2) / dim ) ) - np.exp( np.sum(np.cos(c * vector)) / dim ) + a + np.exp(1)
        return result

    def evaluate(self, cat_sample, cont_sample=None):
        """
        Evaluate function with categorical and continuous parts
        cat_sample: categorical variables (0-num_opts-1, mapped to discrete points in continuous space)
        cont_sample: continuous variables (already in continuous space)
        """
        # Create complete vector
        vector = np.zeros(self.dim)

        # Process categorical variables: directly divide continuous space into num_opts equal points
        if self.n_categorical > 0 and cat_sample is not None:
            for i, cat_val in enumerate(cat_sample):
                # Map discrete values (0,1,2,...,num_opts-1) to equal points in [-32.768, 32.768]
                vector[i] = -32.768 + (65.536 / (self.num_opts - 1)) * cat_val

        # Process continuous variables
        if self.n_continuous > 0 and cont_sample is not None:
            for i, cont_val in enumerate(cont_sample):
                vector[self.n_categorical + i] = cont_val

        return self.ackley(vector)

    def compute(self, X, normalize=False, minimize=True):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        results = []
        for x in X:
            # Split input into categorical and continuous parts
            if self.n_categorical > 0 and self.n_continuous > 0:
                cat_sample = np.array([int(val) for val in x[:self.n_categorical]])
                cont_sample = np.array([float(val) for val in x[self.n_categorical:]])
            elif self.n_categorical > 0:
                cat_sample = np.array([int(val) for val in x])
                cont_sample = None
            else:  # only continuous
                cat_sample = None
                cont_sample = np.array([float(val) for val in x])

            result = self.evaluate(cat_sample, cont_sample)
            results.append(result)

        results = np.array(results)

        if normalize:
            results = (results - np.mean(results)) / (np.std(results) + 1e-8)

        return results.reshape(-1, 1)

    def sample_normalize(self, size=None):
        """Sample normalization for Ackley function"""
        if size is None:
            size = 2 * self.dim + 1
        y = []
        for i in range(size):
            # Generate random categorical variables
            if self.n_categorical > 0:
                x_cat = np.array([np.random.choice(self.config[j]) for j in range(self.n_categorical)])
            else:
                x_cat = np.array([])

            # Generate random continuous variables
            if self.n_continuous > 0:
                x_cont = np.random.uniform(self.lb, self.ub)
            else:
                x_cont = np.array([])

            # Combine and evaluate
            x = np.concatenate([x_cat, x_cont]) if len(x_cat) > 0 and len(x_cont) > 0 else (x_cat if len(x_cat) > 0 else x_cont)
            y.append(self.compute(x, normalize=False))

        y = np.array(y)
        return np.mean(y), np.std(y)

    def get_cocabo_bounds(self):
        bounds = []

        # Add categorical variables
        for i, var_name in enumerate(self.cat_var):
            domain = tuple(range(int(self.n_vertices[i])))
            bounds.append({
                'name': var_name,
                'type': 'categorical',
                'domain': domain
            })

        # Add continuous variables
        for i, var_name in enumerate(self.cont_var):
            domain = (float(self.lb[i]), float(self.ub[i]))
            bounds.append({
                'name': var_name,
                'type': 'continuous',
                'domain': domain
            })

        return bounds


#=========================================================================

class Rosenbrock_benchmark(TestFunction):
    """
    Rosenbrock function for mixed categorical and continuous variables
    Categorical variables: discretized continuous space into num_opts bins mapped to [-5, 10]
    Continuous variables: in [-5, 10]
    """
    problem_type = 'mixed'

    def __init__(self, n_categorical=1, n_continuous=1, num_opts=21, lamda=1e-6, normalize=False, seed=None):
        super().__init__(normalize=normalize)
        self.seed = seed
        self.normalize = normalize
        self.lamda = lamda
        self.n_categorical = n_categorical
        self.n_continuous = n_continuous
        self.num_opts = num_opts  # Number of options for categorical variables

        # Set dimension information
        self.dim = n_categorical + n_continuous
        self.categorical_dims = np.arange(n_categorical) if n_categorical > 0 else np.array([])
        self.continuous_dims = np.arange(n_categorical, self.dim) if n_continuous > 0 else np.array([])

        # Set options for categorical variables
        self.n_vertices = np.array([num_opts] * n_categorical) if n_categorical > 0 else np.array([])
        self.config = self.n_vertices

        # Set variable names
        self.cat_var = [f'x_{i}' for i in range(n_categorical)] if n_categorical > 0 else []
        self.cont_var = [f'x_{i}' for i in range(n_categorical, self.dim)] if n_continuous > 0 else []

        # Set bounds for continuous variables
        if n_continuous > 0:
            self.cont_bounds = [(-5, 10)] * n_continuous
            self.lb = np.array([b[0] for b in self.cont_bounds])
            self.ub = np.array([b[1] for b in self.cont_bounds])
        else:
            self.cont_bounds = []
            self.lb = np.array([])
            self.ub = np.array([])
        # Initialize encoder
        self.encoder = CategoricalEncoder()

        # Set encoding information for benchmark function (simulate to_cat call)
        if self.n_categorical > 0:

            virtual_cat_data = {}
            for var_name in self.cat_var:

                virtual_cat_data[var_name] = list(range(self.num_opts))

            virtual_df = pd.DataFrame(virtual_cat_data)
            self.encoder.to_cat(virtual_df)
    def rosenbrock(self, vector):
        result = 0.
        for i in range(len(vector)-1):
            result += 100 * (vector[i+1] - vector[i]**2)**2 + (1 - vector[i])**2
        return result

    def evaluate(self, cat_sample, cont_sample=None):
        """
        Evaluate function with categorical and continuous parts
        cat_sample: categorical variables (0-num_opts-1, mapped to discrete points in continuous space)
        cont_sample: continuous variables (already in continuous space)
        """
        # Create complete vector
        vector = np.zeros(self.dim)

        # Process categorical variables: directly divide continuous space into num_opts equal points
        if self.n_categorical > 0 and cat_sample is not None:
            for i, cat_val in enumerate(cat_sample):
                # Map discrete values (0,1,2,...,num_opts-1) to equal points in [-5, 10]
                vector[i] = 15 * cat_val / float(self.num_opts - 1) + 5

        # Process continuous variables
        if self.n_continuous > 0 and cont_sample is not None:
            for i, cont_val in enumerate(cont_sample):
                vector[self.n_categorical + i] = cont_val

        return self.rosenbrock(vector)

    def compute(self, X, normalize=False, minimize=True):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        results = []
        for x in X:
            # Split input into categorical and continuous parts
            if self.n_categorical > 0 and self.n_continuous > 0:
                cat_sample = np.array([int(val) for val in x[:self.n_categorical]])
                cont_sample = np.array([float(val) for val in x[self.n_categorical:]])
            elif self.n_categorical > 0:
                cat_sample = np.array([int(val) for val in x])
                cont_sample = None
            else:  # only continuous
                cat_sample = None
                cont_sample = np.array([float(val) for val in x])

            result = self.evaluate(cat_sample, cont_sample)
            results.append(result)

        results = np.array(results)

        if normalize:
            results = (results - np.mean(results)) / (np.std(results) + 1e-8)

        return results.reshape(-1, 1)

    def get_cocabo_bounds(self):
        bounds = []

        # Add categorical variables
        for i, var_name in enumerate(self.cat_var):
            domain = tuple(range(int(self.n_vertices[i])))
            bounds.append({
                'name': var_name,
                'type': 'categorical',
                'domain': domain
            })

        # Add continuous variables
        for i, var_name in enumerate(self.cont_var):
            domain = (float(self.lb[i]), float(self.ub[i]))
            bounds.append({
                'name': var_name,
                'type': 'continuous',
                'domain': domain
            })

        return bounds


#=========================================================================

class Schwefel_benchmark(TestFunction):
    """
    Schwefel function for mixed categorical and continuous variables
    Categorical variables: discretized continuous space into num_opts bins mapped to [-500, 500]
    Continuous variables: in [-500, 500]
    """
    problem_type = 'mixed'

    def __init__(self, n_categorical=1, n_continuous=1, num_opts=21, lamda=1e-6, normalize=False, seed=None):
        super().__init__(normalize=normalize)
        self.seed = seed
        self.normalize = normalize
        self.lamda = lamda
        self.n_categorical = n_categorical
        self.n_continuous = n_continuous
        self.num_opts = num_opts  # Number of options for categorical variables

        # Set dimension information
        self.dim = n_categorical + n_continuous
        self.categorical_dims = np.arange(n_categorical) if n_categorical > 0 else np.array([])
        self.continuous_dims = np.arange(n_categorical, self.dim) if n_continuous > 0 else np.array([])

        # Set options for categorical variables
        self.n_vertices = np.array([num_opts] * n_categorical) if n_categorical > 0 else np.array([])
        self.config = self.n_vertices

        # Set variable names
        self.cat_var = [f'x_{i}' for i in range(n_categorical)] if n_categorical > 0 else []
        self.cont_var = [f'x_{i}' for i in range(n_categorical, self.dim)] if n_continuous > 0 else []

        # Set bounds for continuous variables
        if n_continuous > 0:
            self.cont_bounds = [(-500, 500)] * n_continuous
            self.lb = np.array([b[0] for b in self.cont_bounds])
            self.ub = np.array([b[1] for b in self.cont_bounds])
        else:
            self.cont_bounds = []
            self.lb = np.array([])
            self.ub = np.array([])
        # Initialize encoder
        self.encoder = CategoricalEncoder()

        # Set encoding information for benchmark function (simulate to_cat call)
        if self.n_categorical > 0:

            virtual_cat_data = {}
            for var_name in self.cat_var:

                virtual_cat_data[var_name] = list(range(self.num_opts))

            virtual_df = pd.DataFrame(virtual_cat_data)
            self.encoder.to_cat(virtual_df)
    def schwefel(self, vector):
        vector = np.asarray(vector, dtype=float)
        result = 0
        for index, element in enumerate(vector):
            result += - element * np.sin(np.sqrt(np.abs(element)))
        result += 418.9829 * len(vector)
        return result

    def evaluate(self, cat_sample, cont_sample=None):
        """
        Evaluate function with categorical and continuous parts
        cat_sample: categorical variables (0-num_opts-1, mapped to discrete points in continuous space)
        cont_sample: continuous variables (already in continuous space)
        """
        # Create complete vector
        vector = np.zeros(self.dim)

        # Process categorical variables: directly divide continuous space into num_opts equal points
        if self.n_categorical > 0 and cat_sample is not None:
            for i, cat_val in enumerate(cat_sample):
                # Map discrete values (0,1,2,...,num_opts-1) to equal points in [-500, 500]
                vector[i] = 1000 * cat_val / float(self.num_opts - 1) - 500

        # Process continuous variables
        if self.n_continuous > 0 and cont_sample is not None:
            for i, cont_val in enumerate(cont_sample):
                vector[self.n_categorical + i] = cont_val

        return self.schwefel(vector)

    def compute(self, X, normalize=False, minimize=True):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        results = []
        for x in X:
            # Split input into categorical and continuous parts
            if self.n_categorical > 0 and self.n_continuous > 0:
                cat_sample = np.array([int(val) for val in x[:self.n_categorical]])
                cont_sample = np.array([float(val) for val in x[self.n_categorical:]])
            elif self.n_categorical > 0:
                cat_sample = np.array([int(val) for val in x])
                cont_sample = None
            else:  # only continuous
                cat_sample = None
                cont_sample = np.array([float(val) for val in x])

            result = self.evaluate(cat_sample, cont_sample)
            results.append(result)

        results = np.array(results)

        if normalize:
            results = (results - np.mean(results)) / (np.std(results) + 1e-8)

        return results.reshape(-1, 1)

    def get_cocabo_bounds(self):
        bounds = []

        # Add categorical variables
        for i, var_name in enumerate(self.cat_var):
            domain = tuple(range(int(self.n_vertices[i])))
            bounds.append({
                'name': var_name,
                'type': 'categorical',
                'domain': domain
            })

        # Add continuous variables
        for i, var_name in enumerate(self.cont_var):
            domain = (float(self.lb[i]), float(self.ub[i]))
            bounds.append({
                'name': var_name,
                'type': 'continuous',
                'domain': domain
            })

        return bounds


#=========================================================================

class Griewank_benchmark(TestFunction):
    """
    Griewank function for mixed categorical and continuous variables
    Categorical variables: discretized continuous space into num_opts bins mapped to [-600, 600]
    Continuous variables: in [-600, 600]
    """
    problem_type = 'mixed'

    def __init__(self, n_categorical=1, n_continuous=1, num_opts=21, lamda=1e-6, normalize=False, seed=None):
        super().__init__(normalize=normalize)
        self.seed = seed
        self.normalize = normalize
        self.lamda = lamda
        self.n_categorical = n_categorical
        self.n_continuous = n_continuous
        self.num_opts = num_opts  # Number of options for categorical variables

        # Set dimension information
        self.dim = n_categorical + n_continuous
        self.categorical_dims = np.arange(n_categorical) if n_categorical > 0 else np.array([])
        self.continuous_dims = np.arange(n_categorical, self.dim) if n_continuous > 0 else np.array([])

        # Set options for categorical variables
        self.n_vertices = np.array([num_opts] * n_categorical) if n_categorical > 0 else np.array([])
        self.config = self.n_vertices

        # Set variable names
        self.cat_var = [f'x_{i}' for i in range(n_categorical)] if n_categorical > 0 else []
        self.cont_var = [f'x_{i}' for i in range(n_categorical, self.dim)] if n_continuous > 0 else []

        # Set bounds for continuous variables
        if n_continuous > 0:
            self.cont_bounds = [(-600, 600)] * n_continuous
            self.lb = np.array([b[0] for b in self.cont_bounds])
            self.ub = np.array([b[1] for b in self.cont_bounds])
        else:
            self.cont_bounds = []
            self.lb = np.array([])
            self.ub = np.array([])
        # Initialize encoder
        self.encoder = CategoricalEncoder()

        # Set encoding information for benchmark function (simulate to_cat call)
        if self.n_categorical > 0:

            virtual_cat_data = {}
            for var_name in self.cat_var:

                virtual_cat_data[var_name] = list(range(self.num_opts))

            virtual_df = pd.DataFrame(virtual_cat_data)
            self.encoder.to_cat(virtual_df)
    def griewank(self, vector):
        vector = np.asarray(vector, dtype=float)
        result = 0
        sum_term = 0
        prod_term = 1
        for index, element in enumerate(vector):
            sum_term += element**2 / 4000
            prod_term *= np.cos(element / np.sqrt(index + 1))
        result = sum_term - prod_term + 1
        return result

    def evaluate(self, cat_sample, cont_sample=None):
        """
        Evaluate function with categorical and continuous parts
        cat_sample: categorical variables (0-num_opts-1, mapped to discrete points in continuous space)
        cont_sample: continuous variables (already in continuous space)
        """
        # Create complete vector
        vector = np.zeros(self.dim)

        # Process categorical variables: directly divide continuous space into num_opts equal points
        if self.n_categorical > 0 and cat_sample is not None:
            for i, cat_val in enumerate(cat_sample):
                # Map discrete values (0,1,2,...,num_opts-1) to equal points in [-600, 600]
                vector[i] = 1200 * cat_val / float(self.num_opts - 1) - 600

        # Process continuous variables
        if self.n_continuous > 0 and cont_sample is not None:
            for i, cont_val in enumerate(cont_sample):
                vector[self.n_categorical + i] = cont_val

        return self.griewank(vector)

    def compute(self, X, normalize=False, minimize=True):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        results = []
        for x in X:
            # Split input into categorical and continuous parts
            if self.n_categorical > 0 and self.n_continuous > 0:
                cat_sample = np.array([int(val) for val in x[:self.n_categorical]])
                cont_sample = np.array([float(val) for val in x[self.n_categorical:]])
            elif self.n_categorical > 0:
                cat_sample = np.array([int(val) for val in x])
                cont_sample = None
            else:  # only continuous
                cat_sample = None
                cont_sample = np.array([float(val) for val in x])

            result = self.evaluate(cat_sample, cont_sample)
            results.append(result)

        results = np.array(results)

        if normalize:
            results = (results - np.mean(results)) / (np.std(results) + 1e-8)

        return results.reshape(-1, 1)

    def get_cocabo_bounds(self):
        bounds = []

        # Add categorical variables
        for i, var_name in enumerate(self.cat_var):
            domain = tuple(range(int(self.n_vertices[i])))
            bounds.append({
                'name': var_name,
                'type': 'categorical',
                'domain': domain
            })

        # Add continuous variables
        for i, var_name in enumerate(self.cont_var):
            domain = (float(self.lb[i]), float(self.ub[i]))
            bounds.append({
                'name': var_name,
                'type': 'continuous',
                'domain': domain
            })

        return bounds


#=========================================================================
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # Test the benchmark functions
    functions = [Ackley_benchmark, Rosenbrock_benchmark, Schwefel_benchmark, Griewank_benchmark]

    for func_class in functions:
        print(f"Testing {func_class.__name__}...")
        try:
            # Test with default parameters
            if func_class.__name__ == 'Ackley_benchmark':
                func = func_class(n_categorical=2, n_continuous=2, num_opts=10)
            else:
                func = func_class(n_categorical=1, n_continuous=1, num_opts=10)

            # Test compute method
            test_input = np.array([[0, 0], [5, 5], [9, 9]])
            result = func.compute(test_input)
            print(f"  Sample computation result shape: {result.shape}")
            print(f"  Sample values: {result.flatten()}")

            # Test bounds
            bounds = func.get_cocabo_bounds()
            print(f"  Number of bounds: {len(bounds)}")
            print(f"  Bounds structure: {bounds[0]}")
            print(f"  {func_class.__name__} test passed!\n")

        except Exception as e:
            print(f"  Error testing {func_class.__name__}: {e}\n")
