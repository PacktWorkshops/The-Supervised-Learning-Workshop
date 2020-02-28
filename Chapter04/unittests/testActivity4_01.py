import unittest
import os
import pandas as pd
from statsmodels.tsa.ar_model import AR

class TestingActivity4_01(unittest.TestCase):
    def setUp(self) -> None:
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

        self.data = pd.read_csv(os.path.join(ROOT_DIR, '..', 'Datasets', 'austin_weather.csv'))

    def test_AR(self):
        self.model = AR(self.data.TempAvgF)
        self.model_fit = self.model.fit()
        self.max_lag = self.model_fit.k_ar
        self.assertEqual(self.max_lag, (23))
        self.params = self.model_fit.params[0:4]
        self.assertEqual(round(self.params[0], 4), (1.9094))
        self.assertEqual(round(self.params[1], 4), (0.9121))


if __name__ == '__main__':
    unittest.main()