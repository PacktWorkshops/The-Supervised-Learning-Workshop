import unittest
import os
import pandas as pd
from sklearn.linear_model import LinearRegression

class TestingExercise3_02(unittest.TestCase):
    def setUp(self) -> None:
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

        self.data = pd.read_csv(os.path.join(ROOT_DIR, '..', 'Datasets', 'synth_temp.csv'))

    def test_lin_reg(self):
        self.data = self.data[self.data['Year'] > 1901].groupby('Year').agg('mean')
        self.data['Year'] = self.data.index
        self.linear_model = LinearRegression(fit_intercept = True)
        self.linear_model.fit(self.data['Year'].values.reshape((-1, 1)), self.data.RgnAvTemp)
        self.r2 = self.linear_model.score(self.data['Year'].values.reshape((-1, 1)), self.data.RgnAvTemp)
        self.assertEqual(round(self.linear_model.coef_[0], 3), (0.024))
        self.assertEqual(round(self.linear_model.intercept_, 3), (-27.887))
        self.assertEqual(round(self.r2, 3), (0.844))

if __name__ == '__main__':
    unittest.main()