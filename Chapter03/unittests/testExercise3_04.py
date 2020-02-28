import unittest
import os
import pandas as pd
from sklearn.linear_model import LinearRegression

class TestingExercise3_04(unittest.TestCase):
    def setUp(self) -> None:
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

        self.data = pd.read_csv(os.path.join(ROOT_DIR, '..', 'Datasets', 'synth_temp.csv'))

    def test_poly_reg(self):
        self.data = self.data[self.data['Year'] > 1901].groupby('Year').agg('mean')
        self.data['Year'] = self.data.index.get_level_values(0)
        self.data['Year2'] = self.data['Year'] ** 2
        self.linear_model = LinearRegression(fit_intercept = True)
        self.linear_model.fit(self.data.loc[:, ['Year', 'Year2']],
                              self.data.RgnAvTemp)
        self.r2 = self.linear_model.score(self.data.loc[:, ['Year', 'Year2']],
                                          self.data.RgnAvTemp)
        self.assertEqual(round(self.r2, 3), (0.931))

if __name__ == '__main__':
    unittest.main()