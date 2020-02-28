import unittest
import os
import pandas as pd

class TestingActivity3_01(unittest.TestCase):
    def setUp(self) -> None:
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

        self.data = pd.read_csv(os.path.join(ROOT_DIR, '..', 'Datasets', 'austin_weather.csv'))

    def test_data(self):
        self.assertEqual(self.data.head(5).iloc[0, 0], ('2013-12-21'))
        self.assertEqual(self.data.head(5).loc[4, 'HumidityAvgPercent'], ('71'))
        self.assertEqual(self.data[self.data.index < 365].rolling(window = 20).mean().loc[364, 'TempAvgF'], (58))

if __name__ == '__main__':
    unittest.main()