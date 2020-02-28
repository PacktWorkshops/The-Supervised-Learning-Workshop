import unittest
import os
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score

class TestingExercise3_05(unittest.TestCase):
    def setUp(self) -> None:
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

        self.data = pd.read_csv(os.path.join(ROOT_DIR, '..', 'Datasets', 'synth_temp.csv'))

    def test_grad_desc(self):
        def h_x(Beta, X):
            return np.dot(Beta, X).flatten()
        def J_beta(pred, true):
            return np.mean((pred - true) ** 2)
        def update(pred, true, X, gamma):
            return gamma * np.sum((true - pred) * X, axis = 1)
        self.data = self.data.loc[self.data.Year > 1901]
        self.data_group_year = self.data.groupby(['Year']).agg({'RgnAvTemp' : 'mean'})
        self.data_group_year['Year'] = self.data_group_year.index
        self.data_group_year = self.data_group_year.rename(columns = {'RgnAvTemp' : 'AvTemp'})
        self.X_min = self.data_group_year.Year.min()
        self.X_range = self.data_group_year.Year.max() - self.data_group_year.Year.min()
        self.Y_min = self.data_group_year.AvTemp.min()
        self.Y_range = self.data_group_year.AvTemp.max() - self.data_group_year.AvTemp.min()
        self.scale_X = (self.data_group_year.Year - self.X_min) / self.X_range
        self.train_X = pd.DataFrame({'X0' : np.ones(self.data_group_year.shape[0]),
                                     'X1' : self.scale_X}).transpose()
        self.train_Y = (self.data_group_year.AvTemp - self.Y_min) / self.Y_range
        np.random.seed(42)
        self.Beta = np.random.randn(2).reshape((1, 2)) * 0.1
        self.gamma = 0.0005
        self.max_epochs = 100
        self.y_pred = h_x(self.Beta, self.train_X)
        self.epochs = []
        self.costs = []
        for self.epoch in range(self.max_epochs):
            self.Beta += update(self.y_pred, self.train_Y, self.train_X, self.gamma)
            self.y_pred = h_x(self.Beta, self.train_X)
            self.cost = J_beta(self.y_pred, self.train_Y)
            self.epochs.append(self.epoch)
            self.costs.append(self.cost)
        self.r2 = r2_score(self.train_Y, self.y_pred)
        self.assertEqual(round(self.r2, 3), (0.549))
        self.assertEqual(round(self.cost, 3), (0.035))

if __name__ == '__main__':
    unittest.main()