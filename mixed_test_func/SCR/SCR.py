
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import os
import sys
from autogluon.tabular import TabularPredictor

# 允许在 mixed_test_func/SCR 目录下直接运行 python SCR.py
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)

from test_funcs.base import TestFunction


import warnings
warnings.filterwarnings('ignore')



class SCR(TestFunction):
    problem_type = 'mixed'

    def __init__(self, lamda=1e-6, normalize=False, seed=None, sep='sep'):
        super().__init__(normalize=normalize)
        self.current_dir = os.path.dirname(__file__)
        self.seed = seed
        self.normalize = normalize
        self.lamda = lamda
        self.sep = sep

        self.get_data()
        self.dim = len(self.categorical_dims) + len(self.continuous_dims)
        self.config = self.n_vertices


        if self.sep == 'sep':
            load_path = os.path.join(self.current_dir, 'AutogluonModels', 'SCR')
        else:
            load_path = os.path.join(self.current_dir, 'AutogluonModels', f'SCR_{self.sep}')

        try:
            
            self.model = TabularPredictor.load(load_path)
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
        X = pd.DataFrame(X, columns=self.cat_var + self.cont_var)
        res = self.model.predict(X)
        res += self.lamda * np.random.rand(*res.shape)
        res = res.to_numpy()
        res = np.minimum(res, 100.0)

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
                                                                                                          presets='medium_quality')
        
        metrics = ["mae", "mse", "rmse", "r2", "pearsonr", "median_absolute_error"]
        leaderboard = predictor.leaderboard(test_data, extra_metrics=metrics, silent=True)
        leaderboard_path =  os.path.join(save_path, "leaderboard_metrics.csv")
        leaderboard.to_csv(leaderboard_path, index=False)
        return predictor

    def get_data(self):
        file_path = os.path.join(self.current_dir, "SCR.csv")
        self.data = pd.read_csv(file_path, index_col=None)

        # 统一 SCR.csv 中带单位/换行的列名为内部标准列名
        self.data = self.data.rename(columns={
            "Fe\nwt %": "Fe",
            "Cu\nwt %": "Cu",
            "Si/Al": "Si_Al",
            "calcination \ntemperature (℃)": "calcination_temp",
            "calcination \ntime (h)": "calcination_time",
            "Aging_O2\n(%)": "Aging_O2",
            "Aging_H2O\n(%)": "Aging_H2O",
            "Aging_CO2\n(%)": "Aging_CO2",
            "Aging_Temp\n(℃)": "Aging_Temp",
            "Aging_Time\n(h)": "Aging_Time",
            "NO\n(ppm)": "NO",
            "NH3\n(ppm)": "NH3",
            "O2\n(%)": "O2",
            "H2O\n(%)": "H2O",
            "CO2\n(%)": "CO2",
            "GHSV\n(10^5 * h^{-1})": "GHSV",
        })
        # self.data['y'] = 2 * self.data["C2H4y"] + self.data["C2H6y"] - self.data["COy"] - 2 * self.data["CO2y"]
        self.data['y'] = self.data['Conversion']
        self.encoder = CategoricalEncoder()
        if self.sep == 'sep':
            self.cat_var = ["species"]
            self.cont_var = ["Fe", "Cu", "Si_Al", "calcination_temp", "calcination_time",
                         "Aging_O2", "Aging_H2O", "Aging_CO2", "Aging_Temp", "Aging_Time",
                         "NO","NH3", "O2", "H2O", "GHSV", "Measurement_Temp"]
        elif self.sep == 'm1':
            self._merge_phase_1()
            self.cat_var = ["m1"]
            self.cont_var = ["Aging_O2", "Aging_H2O", "Aging_CO2", "Aging_Temp", "Aging_Time",
                             "NO","NH3", "O2", "H2O", "GHSV", "Measurement_Temp"]
        elif self.sep == 'm12':
            self._merge_phase_12()
            self.cat_var = ["m12"]
            self.cont_var = ["NO","NH3", "O2", "H2O", "GHSV", "Measurement_Temp"]
        elif self.sep == 'm1m2':
            self._merge_phase_1_2()
            self.cat_var = ["m1", "m2"]
            self.cont_var = ["NO","NH3", "O2", "H2O", "GHSV", "Measurement_Temp"]
        else:
            raise ValueError('Unknown type for handling Phase-I reaction')

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

    def _merge_phase_1(self):
        required_cols = ["Fe", "Cu", "Si_Al", "species", "calcination_temp", "calcination_time"]
        missing_cols = [col for col in required_cols if col not in self.data.columns]

        if missing_cols:
            raise ValueError(f"Missing necessary columns: {missing_cols}")

        self.data['m1'] = (
                self.data['Fe'].astype(str) + '|' + self.data['Cu'].astype(str) + '|' +
                self.data['Si_Al'].astype(str) + '|' + self.data['species'].astype(str) + '|' +
                self.data['calcination_temp'].astype(str) + '|' + self.data['calcination_time'].astype(str)
        )

    def _merge_phase_12(self):
        required_cols = ["Fe", "Cu", "Si_Al", "species",  "calcination_temp", "calcination_time",
                         "Aging_O2", "Aging_H2O", "Aging_CO2", "Aging_Temp", "Aging_Time"]
        missing_cols = [col for col in required_cols if col not in self.data.columns]

        if missing_cols:
            raise ValueError(f"Missing necessary columns: {missing_cols}")

        self.data['m12'] = (
                self.data['Fe'].astype(str) + '|' + self.data['Cu'].astype(str) + '|' +
                self.data['Si_Al'].astype(str) + '|' + self.data['species'].astype(str) + '|' +
                self.data['calcination_temp'].astype(str) + '|' + self.data['calcination_time'].astype(str) + '|' +
                self.data['Aging_O2'].astype(str) + '|' + self.data['Aging_H2O'].astype(str) + '|' +
                self.data['Aging_CO2'].astype(str) + '|' +
                self.data['Aging_Temp'].astype(str) + '|' + self.data['Aging_Time'].astype(str)
        )

    def _merge_phase_1_2(self):
        required_cols = ["Fe", "Cu", "Si_Al", "species",  "calcination_temp", "calcination_time",
                         "Aging_O2", "Aging_H2O", "Aging_CO2", "Aging_Temp", "Aging_Time"]
        missing_cols = [col for col in required_cols if col not in self.data.columns]

        if missing_cols:
            raise ValueError(f"Missing necessary columns: {missing_cols}")

        self.data['m1'] = (
                self.data['Fe'].astype(str) + '|' + self.data['Cu'].astype(str) + '|' +
                self.data['Si_Al'].astype(str) + '|' + self.data['species'].astype(str) + '|' +
                self.data['calcination_temp'].astype(str) + '|' + self.data['calcination_time'].astype(str)
        )
        self.data['m2'] = (
                self.data['Aging_O2'].astype(str) + '|' + self.data['Aging_H2O'].astype(str) + '|' +
                self.data['Aging_CO2'].astype(str) + '|' +
                self.data['Aging_Temp'].astype(str) + '|' + self.data['Aging_Time'].astype(str)
        )


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
    model = SCR(normalize=False, lamda=1e-6, seed=1)
