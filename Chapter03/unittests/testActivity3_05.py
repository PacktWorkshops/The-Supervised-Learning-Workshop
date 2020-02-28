import unittest
import os
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.linear_model import SGDRegressor

class TestingActivity3_05(unittest.TestCase):
    def setUp(self) -> None:
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

        self.data = pd.read_csv(os.path.join(ROOT_DIR, '..', 'Datasets', 'austin_weather.csv'))

    def test_SGD(self):
        self.data = self.data.loc[:, ['Date', 'TempAvgF']]
        self.data.loc[:, 'Year'] = self.data.loc[:, 'Date'].str.slice(0, 4).astype('int')
        self.data.loc[:, 'Month'] = self.data.loc[:, 'Date'].str.slice(5, 7).astype('int')
        self.data.loc[:, 'Day'] = self.data.loc[:, 'Date'].str.slice(8, 10).astype('int')
        self.data['20_d_mov_avg'] = self.data.TempAvgF.rolling(20).mean()
        self.data_one_year = self.data.loc[self.data.Year == 2015, :].reset_index()
        self.data_one_year['Day_of_Year'] = self.data_one_year.index + 1
        self.X_min = self.data_one_year.Day_of_Year.min()
        self.X_range = self.data_one_year.Day_of_Year.max() - self.data_one_year.Day_of_Year.min()
        self.Y_min = self.data_one_year.TempAvgF.min()
        self.Y_range = self.data_one_year.TempAvgF.max() - self.data_one_year.TempAvgF.min()
        self.scale_X = (self.data_one_year.Day_of_Year - self.X_min) / self.X_range
        self.train_X = self.scale_X.ravel()
        self.train_Y = ((self.data_one_year.TempAvgF - self.Y_min) / self.Y_range).ravel()
        np.random.seed(42)
        self.model = SGDRegressor(loss = 'squared_loss',
                                    max_iter = 100,
                                    learning_rate = 'constant',
                                    eta0 = 0.0005,
                                    tol = 0.00009,
                                    penalty = 'none')
        self.model.fit(self.train_X.reshape((-1, 1)), self.train_Y)
        self.Beta0 = (self.Y_min + self.Y_range * self.model.intercept_[0] -
                self.Y_range * self.model.coef_[0] * self.X_min / self.X_range)
        self.Beta1 = self.Y_range * self.model.coef_[0] / self.X_range
        self.pred_X = self.data_one_year['Day_of_Year']
        self.pred_Y = self.model.predict(self.train_X.reshape((-1, 1)))
        self.r2 = r2_score(self.train_Y, self.pred_Y)
        self.assertEqual(round(self.Beta0, 1), (61.5))
        self.assertEqual(round(self.Beta1, 3), (0.045))
        self.assertEqual(round(self.r2, 3), (0.095))

if __name__ == '__main__':
    unittest.main()