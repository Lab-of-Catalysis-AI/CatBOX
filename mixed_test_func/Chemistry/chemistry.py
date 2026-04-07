
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import os
from autogluon.tabular import TabularPredictor
from test_funcs.base import TestFunction


import warnings
warnings.filterwarnings('ignore')



class Chemistry(TestFunction):
    problem_type = 'mixed'
    # OCM returns -fx
    def __init__(self, lamda=1e-6, normalize=False, seed=None, sep='atom', prob='OCM2'):
        super().__init__(normalize=normalize)
        self.sep = sep
        self.current_dir = os.path.dirname(__file__)
        self.seed = seed
        self.normalize = normalize
        self.lamda = lamda
        self.prob = prob

        self.get_data()
        self.dim = len(self.categorical_dims) + len(self.continuous_dims)
        self.config = self.n_vertices

        # true_atom and all_update_true use sep model but with different variable formats
        model_sep = 'sep' if self.sep in ['true_atom', 'all_update_true'] else self.sep
        load_path = os.path.join(self.current_dir, 'AutogluonModels', f'{self.prob}_{model_sep}')

        try:
            self.model = TabularPredictor.load(load_path, require_version_match=False, require_py_version_match=False)
        except FileNotFoundError:
            self.model = self.create_model(load_path)

        if self.normalize:
            self.mean, self.std = self.sample_normalize()
        else:
            self.mean, self.std = None, None

        # self._mercer_feature()

    def parse_row(self, row):
        parts = row['merged_3'].split('|')

        M1 = M2 = M3 = Support = None

        # Assign fields based on self.sep value
        if self.sep == 'M1':
            M1 = row['M1']  # Assume M1 column exists (if self.sep is column name)
            M2, M3, Support = parts
        elif self.sep == 'M2':
            M2 = row['M2']  # Assume M2 column exists
            M1, M3, Support = parts
        elif self.sep == 'M3':
            M3 = row['M3']  # Assume M3 column exists
            M1, M2, Support = parts
        elif self.sep == 'Support':
            Support = row['Support']  # Assume Support column exists
            M1, M2, M3 = parts

        return M1, M2, M3, Support

    def compute(self, X, normalize=False, minimize=True):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        X = pd.DataFrame(X, columns=self.cat_var + self.cont_var)
        X1 = self.encoder.from_cat(X[self.cat_var].astype(int))
        if self.sep == 'sep':
            for idx, row in X1.iterrows():
                M1, M2, M3, Support = row['M1'], row['M2'], row['M3'], row['Support']
                if M1 == 'n.a.':
                    X.at[idx, 'M1_mol'] = 0
                if M2 == 'n.a.':
                    X.at[idx, 'M2_mol'] = 0
                if M3 == 'n.a.':
                    X.at[idx, 'M3_mol'] = 0
        elif self.sep in ['M1', 'M2', 'M3', 'Support']:
            for idx, row in X1.iterrows():
                M1, M2, M3, Support = self.parse_row(row)
                if M1 == 'n.a.':
                    X.at[idx, 'M1_mol'] = 0
                if M2 == 'n.a.':
                    X.at[idx, 'M2_mol'] = 0
                if M3 == 'n.a.':
                    X.at[idx, 'M3_mol'] = 0

        elif self.sep == 'atom':
            split_data = X1['merged_4'].str.split('|', expand=True)
            M1 = split_data[0]  # Part 1 (M1)
            M2 = split_data[1]  # Part 2 (M2)
            M3 = split_data[2]  # Part 3 (M3)

            # 2. Update M1_mol: if M1 == "n.a.", set to 0, otherwise keep original value
            X.loc[M1 == 'n.a.', 'M1_mol'] = 0
            X.loc[M2 == 'n.a.', 'M2_mol'] = 0
            X.loc[M3 == 'n.a.', 'M3_mol'] = 0
        elif self.sep == 'true_atom':
            merged_4_strings = X1['merged_4']
            
            # Step 2: Split the strings to get separate variables
            split_data = merged_4_strings.str.split('|', expand=True)
            M1 = split_data[0]  # Part 1 (M1)
            M2 = split_data[1]  # Part 2 (M2)
            M3 = split_data[2]  # Part 3 (M3)
            Support = split_data[3]  # Part 4 (Support)
            
            # Step 3: Create X_sep with correct column order for sep model
            X_sep = pd.DataFrame(index=X.index)
            
            # Step 4: Encode categorical variables using sep_encoder
            X_sep['M1'] = self.sep_encoder.encoders['M1'].transform(M1.astype(str))
            X_sep['M2'] = self.sep_encoder.encoders['M2'].transform(M2.astype(str))
            X_sep['M3'] = self.sep_encoder.encoders['M3'].transform(M3.astype(str))
            X_sep['Support'] = self.sep_encoder.encoders['Support'].transform(Support.astype(str))
            
            # Step 5: Add continuous variables in the same order as sep mode
            X_sep['M1_mol'] = X['M1_mol']
            X_sep['M2_mol'] = X['M2_mol']
            X_sep['M3_mol'] = X['M3_mol']
            X_sep['Temp'] = X['Temp']
            X_sep['Ar_flow'] = X['Ar_flow']
            X_sep['CH4_flow'] = X['CH4_flow']
            X_sep['O2_flow'] = X['O2_flow']
            X_sep['CT'] = X['CT']
            
            # Step 6: Use sep format for prediction
            res = self.model.predict(X_sep)
            res += self.lamda * np.random.rand(*res.shape)
            res = res.to_numpy()
            
            if normalize:
                res = (res - self.mean) / self.std
            return -res.reshape(-1, 1)
        
        elif self.sep == 'all_update_true':
            # Parse merged_all format: M1_M1_mol|M2_M2_mol|M3_M3_mol|Support
            merged_all_strings = X1['merged_all']
            
            # Split by '|' to get 4 parts: M1_M1_mol, M2_M2_mol, M3_M3_mol, Support
            split_data = merged_all_strings.str.split('|', expand=True)
            
            # Extract M1, M1_mol from first part (format: M1_M1_mol)
            m1_parts = split_data[0].str.split('_', expand=True)
            M1 = m1_parts[0]
            M1_mol = pd.to_numeric(m1_parts[1], errors='coerce')
            
            # Extract M2, M2_mol from second part
            m2_parts = split_data[1].str.split('_', expand=True)
            M2 = m2_parts[0]
            M2_mol = pd.to_numeric(m2_parts[1], errors='coerce')
            
            # Extract M3, M3_mol from third part
            m3_parts = split_data[2].str.split('_', expand=True)
            M3 = m3_parts[0]
            M3_mol = pd.to_numeric(m3_parts[1], errors='coerce')
            
            # Extract Support from fourth part
            Support = split_data[3]
            
            # Handle n.a. cases
            M1_mol.loc[M1 == 'n.a.'] = 0
            M2_mol.loc[M2 == 'n.a.'] = 0
            M3_mol.loc[M3 == 'n.a.'] = 0
            
            # Create X_sep with correct column order for sep model
            X_sep = pd.DataFrame(index=X.index)
            
            # Encode categorical variables using sep_encoder
            X_sep['M1'] = self.sep_encoder.encoders['M1'].transform(M1.astype(str))
            X_sep['M2'] = self.sep_encoder.encoders['M2'].transform(M2.astype(str))
            X_sep['M3'] = self.sep_encoder.encoders['M3'].transform(M3.astype(str))
            X_sep['Support'] = self.sep_encoder.encoders['Support'].transform(Support.astype(str))
            
            # Add continuous variables (M1_mol, M2_mol, M3_mol extracted from merged_all, others from X)
            X_sep['M1_mol'] = M1_mol
            X_sep['M2_mol'] = M2_mol
            X_sep['M3_mol'] = M3_mol
            X_sep['Temp'] = X['Temp']
            X_sep['Ar_flow'] = X['Ar_flow']
            X_sep['CH4_flow'] = X['CH4_flow']
            X_sep['O2_flow'] = X['O2_flow']
            X_sep['CT'] = X['CT']
            
            # Use sep format for prediction
            res = self.model.predict(X_sep)
            res += self.lamda * np.random.rand(*res.shape)
            res = res.to_numpy()
            
            if normalize:
                res = (res - self.mean) / self.std
            return -res.reshape(-1, 1)

        res = self.model.predict(X)
        res += self.lamda * np.random.rand(*res.shape)
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
        from sklearn.model_selection import train_test_split
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
        if self.prob == 'OCM2':
            if self.sep in ['all_update', 'all_update_true']:
                file_path = os.path.join(self.current_dir, "OCM_data_with_performance2_with_name_update.csv")
            else:
                file_path = os.path.join(self.current_dir, "OCM_data_with_performance2.csv")
        elif self.prob == 'OCM1':
            file_path = os.path.join(self.current_dir, "OCM_data_with_performance1.csv")

        self.data = pd.read_csv(file_path, index_col=None)
        if self.sep in ['all_update', 'all_update_true']:
            drop_cols = [col for col in ['Name', 'Unnamed: 0'] if col in self.data.columns]
            if drop_cols:
                self.data = self.data.drop(columns=drop_cols)
        self.data['y'] = self.data['Performance']
        self.encoder = CategoricalEncoder()

        if self.sep == 'sep':
            self.cat_var = ["M1", "M2", "M3", "Support"]
            self.cont_var = ["M1_mol", "M2_mol", "M3_mol", "Temp", "Ar_flow", "CH4_flow", "O2_flow", "CT"]
        elif self.sep in ['M1', 'M2', 'M3', 'Support']:
            self._merge_c41()
            self.cat_var = [self.sep, "merged_3"]
            self.cont_var = ["M1_mol", "M2_mol", "M3_mol", "Temp", "Ar_flow", "CH4_flow", "O2_flow", "CT"]
        elif self.sep == 'atom':
            self._merge_atom()
            self.cat_var = ["merged_4"]
            self.cont_var = ["M1_mol", "M2_mol", "M3_mol", "Temp", "Ar_flow", "CH4_flow", "O2_flow", "CT"]
        elif self.sep == 'atom-mol':
            self._merge_atom_mol_to_discrete()
            self.cat_var = ["M1_merged", "M2_merged", "M3_merged", "Support"]
            self.cont_var = ["Temp", "Ar_flow", "CH4_flow", "O2_flow", "CT"]
        elif self.sep == 'all' or self.sep == 'all_update':
            self._merge_all_to_discrete()
            self.cat_var = ["merged_all"]
            self.cont_var = ["Temp", "Ar_flow", "CH4_flow", "O2_flow", "CT"]
        elif self.sep == 'true_atom':
            self._merge_atom()
            # For optimization: use merged_4 (atom format)
            self.cat_var = ["merged_4"]
            self.cont_var = ["M1_mol", "M2_mol", "M3_mol", "Temp", "Ar_flow", "CH4_flow", "O2_flow", "CT"]
            # For model prediction: need separate variables (sep format)
            self.sep_cat_var = ["M1", "M2", "M3", "Support"]
        elif self.sep == 'all_update_true':
            self._merge_all_to_discrete()
            # For optimization: use merged_all (all format with M1_M1_mol|M2_M2_mol|M3_M3_mol|Support)
            self.cat_var = ["merged_all"]
            self.cont_var = ["Temp", "Ar_flow", "CH4_flow", "O2_flow", "CT"]
            # For model prediction: need separate variables (sep format)
            self.sep_cat_var = ["M1", "M2", "M3", "Support"]
        else:
            raise ValueError(f'Unknown type {self.sep} for handling Phase-I reaction')


        self.x = pd.concat([self.encoder.to_cat(self.data[self.cat_var]), self.data[self.cont_var]], axis=1)
        self.y = self.data[['y']]
        self.categorical_dims = np.arange(0,len(self.cat_var))
        self.continuous_dims = np.arange(len(self.cat_var),len(self.cont_var)+len(self.cat_var))
        self.n_vertices = [self.x[i].max()+1 for i in self.cat_var]
        self.n_vertices = np.array(self.n_vertices)
        self.lb = np.array([self.x[i].min() for i in self.cont_var])
        self.ub = np.array([self.x[i].max() for i in self.cont_var])
        
        # For true_atom and all_update_true modes: create separate encoder for sep format variables
        if self.sep in ['true_atom', 'all_update_true']:
            self.sep_encoder = CategoricalEncoder()
            self.sep_encoder.to_cat(self.data[self.sep_cat_var])

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

    def _merge_atom_mol_to_discrete(self):
        for i in range(1, 4):
            str_col = f'M{i}'  # String column name
            mol_col = f'M{i}_mol'  # Numeric column name
            if str_col not in self.data.columns or mol_col not in self.data.columns:
                continue
            new_col_name = f'M{i}_merged'
            self.data[new_col_name] = self.data[str_col].astype(str) + '_' + self.data[mol_col].astype(str)

    def _merge_all_to_discrete(self):
        required_cols = ['M1', 'M2', 'M3', 'Support','M1_mol', 'M2_mol', 'M3_mol']
        missing_cols = [col for col in required_cols if col not in self.data.columns]

        if missing_cols:
            raise ValueError(f"Missing necessary columns: {missing_cols}")

        self.data['merged_all'] = (
                self.data['M1'].astype(str) + '_' + self.data['M1_mol'].astype(str) + '|' +
                self.data['M2'].astype(str) + '_' + self.data['M2_mol'].astype(str) + '|' +
                self.data['M3'].astype(str) + '_' + self.data['M3_mol'].astype(str) + '|' +
                self.data['Support'].astype(str)
        )

    def _merge_c41(self):
        dis_cols = ['M1', 'M2', 'M3', 'Support']
        dis_remain_cols =  [col for col in dis_cols if col != self.sep]
        self.data['merged_3'] = self.data[dis_remain_cols].astype(str).apply('|'.join, axis=1)

    def _merge_atom(self):
        dis_cols = ['M1', 'M2', 'M3', 'Support']
        self.data['merged_4'] = self.data[dis_cols].astype(str).apply('|'.join, axis=1)



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
    Chemistry(normalize=False, lamda=1e-6, seed=1, sep='atom', prob='OCM2')
