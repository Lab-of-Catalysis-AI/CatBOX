
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import os
from autogluon.tabular import TabularPredictor
from test_funcs.base import TestFunction


import warnings
warnings.filterwarnings('ignore')



class DAR(TestFunction):
    problem_type = 'mixed'

    def __init__(self, lamda=1e-6, normalize=False, seed=None, sep='normal'):
        super().__init__(normalize=normalize)
        self.current_dir = os.path.dirname(__file__)
        self.seed = seed
        self.normalize = normalize
        self.lamda = lamda
        self.sep = sep

        self.get_data()
        self.dim = len(self.categorical_dims) + len(self.continuous_dims)
        self.config = self.n_vertices

        if self.sep != 'normal':
            raise ValueError("Unsupported DAR mode. Only 'normal' is supported.")

        load_path = os.path.join(self.current_dir, 'AutogluonModels', 'DAR_medium')
        print('load_path:', load_path)
        try:
            self.model = TabularPredictor.load(load_path, require_version_match=False, require_py_version_match=False)
            self.feature_columns = None
        except FileNotFoundError:
            self.model = self.create_model(load_path)

        if self.normalize:
            self.mean, self.std = self.sample_normalize()
        else:
            self.mean, self.std = None, None

        # self._mercer_feature()

    def compute(self, X, normalize=False, minimize=True):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # Separate categorical variables (integer encoded) and continuous variables
        num_cat = len(self.cat_var)
        X_cat_encoded = X[:, :num_cat].astype(int)
        X_cont = X[:, num_cat:]
        
        # Convert categorical variables from integer encoding back to strings (SMILES)
        X_cat_df = pd.DataFrame(X_cat_encoded, columns=self.cat_var)
        X_cat_decoded = self.encoder.from_cat(X_cat_df)
        
        # Combine categorical variables (strings) and continuous variables
        X_cont_df = pd.DataFrame(X_cont, columns=self.cont_var)
        X = pd.concat([X_cat_decoded, X_cont_df], axis=1)

        res = self.model.predict(X)

        # Add noise for all cases
        res += self.lamda * np.random.rand(*res.shape)

        # Convert to numpy array if not already
        if not isinstance(res, np.ndarray):
            res = res.to_numpy()

        if normalize:
            res = (res - self.mean) / self.std
        return -res.reshape(-1, 1)

    def sample_normalize(self, size=None):
        from cas.localbo_utils import latin_hypercube, from_unit_cube
        if size is None:
            size = 2 * self.dim + 1
        y = []
        for i in range(size):
            x_cat = np.array([np.random.choice(self.config[_]) for _ in range(self.categorical_dims.shape[0])])
            x_cont = latin_hypercube(1, self.continuous_dims.shape[0])
            x_cont = from_unit_cube(x_cont, self.lb, self.ub).flatten()
            x = np.hstack((x_cat, x_cont))
            y.append(self.compute(x, normalize=False))
        y = np.array(y)
        return np.mean(y), np.std(y)

    def create_model(self, save_path):
        df = pd.concat([self.x, self.y], axis=1)
        for i in self.cat_var:
            df[i] = df[i].astype('category')
        for j in self.cont_var:
            df[j] = df[j].astype('float')
        train_test_rate = 0.7
        train_data = df.sample(n=int(train_test_rate * len(df)))
        test_data = df.drop(train_data.index)
        predictor = TabularPredictor(label='y', eval_metric='mean_absolute_error', path=save_path).fit(train_data,
                                                                                                          time_limit=600,
                                                                                                          presets='best_quality')
        leaderboard = predictor.leaderboard(test_data, extra_metrics=['mae'], silent=True)
        print(leaderboard[['model', 'score_val', 'eval_metric']].set_index('model'))
        return predictor

    def get_data(self):
        file_path = os.path.join(self.current_dir, "DAR.csv")
        self.data = pd.read_csv(file_path, index_col=None)
        self.data['y'] = self.data['yield']
        self.encoder = CategoricalEncoder()


        self.cat_var = ["base_SMILES","ligand_SMILES","solvent_SMILES"]
        self.cont_var = ["concentration","temperature"]


        self.x = pd.concat([self.encoder.to_cat(self.data[self.cat_var]), self.data[self.cont_var]], axis=1)
        self.y = self.data[['y']]
        self.categorical_dims = np.arange(0,len(self.cat_var))
        self.continuous_dims = np.arange(len(self.cat_var),len(self.cont_var)+len(self.cat_var))
        self.n_vertices = [self.x[i].max()+1 for i in self.cat_var]
        self.n_vertices = np.array(self.n_vertices)
        self.lb = np.array([self.x[i].min() for i in self.cont_var])
        self.ub = np.array([self.x[i].max() for i in self.cont_var])

    def get_cocabo_bounds(self):
        bounds = []

        for i, var_name in enumerate(self.cat_var):
            domain = tuple(range(int(self.n_vertices[i])))
            bounds.append({
                'name': var_name,
                'type': 'categorical',
                'domain': domain
            })

        for i, var_name in enumerate(self.cont_var):
            domain = (float(self.lb[i]), float(self.ub[i]))  # (min_val, max_val)
            bounds.append({
                'name': var_name,
                'type': 'continuous',
                'domain': domain
            })

        return bounds


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

if __name__ == "__main__":
    DAR(normalize=False, lamda=1e-6, seed=1)
