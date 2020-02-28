import unittest
import os
import pandas as pd
from sklearn.linear_model import LinearRegression

class TestingExercise3_03(unittest.TestCase):
    def setUp(self) -> None:
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

        self.data = pd.read_csv(os.path.join(ROOT_DIR, '..', 'Datasets', 'synth_temp.csv'))

    def test_dummies_lin_reg(self):
        self.data = self.data[self.data['Year'] > 1901].groupby(['Year', 'Region']).agg('mean')
        self.data['Region'] = self.data.index.get_level_values(1)
        self.data['Year'] = self.data.index.get_level_values(0)
        self.data = self.data.droplevel(0, axis = 0).reset_index(drop = True)
        self.dummy_cols = pd.get_dummies(self.data.Region, drop_first = True)
        self.data = pd.concat([self.data, self.dummy_cols], axis = 1)
        self.linear_model = LinearRegression(fit_intercept = True)
        self.linear_model.fit(self.data.loc[:, 'Year':'L'],
                              self.data.RgnAvTemp)
        self.r2 = self.linear_model.score(self.data.loc[:, 'Year':'L'],
                                          self.data.RgnAvTemp)
        self.assertEqual(round(self.r2, 3), (0.778))

if __name__ == '__main__':
    unittest.main()