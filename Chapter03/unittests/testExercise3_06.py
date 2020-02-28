import unittest
import os
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.linear_model import SGDRegressor

class TestingExercise3_06(unittest.TestCase):
    def setUp(self) -> None:
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

        self.data = pd.read_csv(os.path.join(ROOT_DIR, '..', 'Datasets', 'synth_temp.csv'))

    def test_SGD(self):
        self.data = self.data.loc[self.data.Year > 1901]
        self.data_group_year = self.data.groupby(['Year']).agg({'RgnAvTemp' : 'mean'})
        self.data_group_year['Year'] = self.data_group_year.index
        self.data_group_year = self.data_group_year.rename(columns = {'RgnAvTemp' : 'AvTemp'})
        self.X_min = self.data_group_year.Year.min()
        self.X_range = self.data_group_year.Year.max() - self.data_group_year.Year.min()
        self.Y_min = self.data_group_year.AvTemp.min()
        self.Y_range = self.data_group_year.AvTemp.max() - self.data_group_year.AvTemp.min()
        self.scale_X = (self.data_group_year.Year - self.X_min) / self.X_range
        self.train_X = self.scale_X.ravel()
        self.train_Y = ((self.data_group_year.AvTemp - self.Y_min) / self.Y_range).ravel()
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
        self.pred_X = self.data_group_year['Year']
        self.pred_Y = self.model.predict(self.train_X.reshape((-1, 1)))
        self.r2 = r2_score(self.train_Y, self.pred_Y)
        self.assertEqual(round(self.r2, 3), (0.544))

if __name__ == '__main__':
    unittest.main()