import gpflow
import numpy as np
import pandas as pd
from warnings import simplefilter
from sklearn.metrics import r2_score

simplefilter(action='ignore', category=FutureWarning)
import warnings

warnings.filterwarnings("ignore")

class GPModel():
    def __init__(self, prior_file='Prior.csv', test_file='Test.csv', batch_size=10) -> None:
        self.kernel = gpflow.kernels.RationalQuadratic() + gpflow.kernels.White() + gpflow.kernels.Matern12() + gpflow.kernels.Matern12() + gpflow.kernels.White() + gpflow.kernels.RationalQuadratic()
        self.opt = gpflow.optimizers.Scipy()
        self.data_file = [prior_file, test_file]
        self.batch_size = batch_size
        self.reset_dataset()
        
    def reset_dataset(self):
        # Extract prior data
        self.prior_data = pd.read_csv(self.data_file[0], delimiter=',')
        
        # Extract test data
        self.test_data = pd.read_csv(self.data_file[1], delimiter=',')
        self.X_test = self.test_data[['X1', 'X2', 'X3']]
        self.y_actual = self.test_data[['y']]
        
        if set(self.X_test.columns.values) != set(['X1', 'X2', 'X3']):
            raise ValueError(f'X_test column names must be [\'X1\', \'X2\', \'X3\'], got {self.X_test.columns.values}')
        
        if self.y_actual.columns.values != ['y']:
            raise ValueError(f'y_actual column names must be [\'y\'], got {self.y_actual.columns.values}')
          
        self.y_actual_np = self.eliminate_zeros(self.y_actual.iloc[:, 0].values)
        
        l = len(self.y_actual)
        self._data_length = l // self.batch_size
        
        if l % self.batch_size:
            self._data_length += 1
        
        self.max_idx = self._data_length - 1
        
    def eliminate_zeros(self, data):
        for i in range(len(data)):
            if data[i] == 0.0:
                data[i] = 1e-5
        return data
    
    def extract_y_data(self, df):
        y = df.iloc[:, 3].values
        y = self.eliminate_zeros(y)
        y = np.atleast_2d(y).flatten()
        y = np.log10(y)
        #Normalizing y
        self.y_std = np.std(y, ddof=1)
        self.y_m = np.mean(y)
        return (y - self.y_m)/self.y_std
    
    def extract_x_data(self, df, prior=True):
        x1 = df.iloc[:, 0].values / 1e5
        x2 = df.iloc[:, 1].values
        x3 = df.iloc[:, 2].values
        
        #Transforming 1D arrays to 2D
        x1 = np.atleast_2d(x1).flatten().reshape(-1,1)
        x2 = np.atleast_2d(x2).flatten().reshape(-1,1)
        x3 = np.atleast_2d(x3).flatten().reshape(-1,1)
        
        x1 = np.log10(x1)
        
        if prior:
            self.x1_m = np.mean(x1)
            self.x1_std = np.std(x1,ddof=1)

            #Extracting the mean and std. dev
            self.x2_m = np.mean(x2)
            self.x2_std = np.std(x2,ddof=1)

            #Extracting the mean and std. dev for bl_test
            self.x3_m = np.mean(x3)
            self.x3_std = np.std(x3,ddof=1)

        #Standardising the input features
        x1_s = (x1 - self.x1_m)/self.x1_std
        x2_s = (x2 - self.x2_m)/self.x2_std
        x3_s = (x3 - self.x3_m)/self.x3_std
        
        x_s = np.vstack((x1_s.flatten(),x2_s.flatten(), x3_s.flatten())).T
        
        return x_s
        
    @property
    def data_length(self):
        return self._data_length
    
    def update_model(self, data_point_idx):
        if data_point_idx is not None:
            self.add_data_point(data_point_idx)
            self._data_length -= 1

        x_s = self.extract_x_data(self.prior_data)
        y_s = self.extract_y_data(self.prior_data)
        self.model = gpflow.models.GPR(
                    data=(x_s, y_s.reshape(-1, 1)),
                    kernel=self.kernel,
                    noise_variance=10**-5)
        gpflow.utilities.set_trainable(self.model.likelihood.variance,False)
        self.opt.minimize(self.model.training_loss, self.model.trainable_variables)
        
    def add_data_point(self, idx):
        start_idx = idx * self.batch_size
        if idx != self.max_idx:
            end_idx = start_idx + self.batch_size
        else:
            end_idx = start_idx + len(self.test_data) % self.batch_size
        # X_data_point = self.X_test[start_idx:end_idx]
        # y_data_point = self.y_actual[start_idx:end_idx]
        # new_rows = pd.concat([X_data_point, y_data_point], axis=1)
        
        new_rows = self.test_data[start_idx:end_idx]
        
        l = len(self.prior_data)
        for i in range(len(new_rows)):
            self.prior_data.loc[l + i] = new_rows.iloc[i]
            
    def add_data_list(self, idx_list):
        for idx in idx_list:
            self.add_data_point(idx)
            
    def calculate_r2(self):
        self.x_test_np = self.extract_x_data(self.X_test, False)      
        y_pred, _ = self.model.predict_f(self.x_test_np)
        y_pred = y_pred.numpy()
        y_pred = 10 ** ((y_pred * self.y_std) + self.y_m)

        # calculate the R2 score
        return r2_score(self.y_actual_np, y_pred)
    
    def write_final_prior(self, final_path):
        self.prior_data.to_csv(final_path, index=False)
