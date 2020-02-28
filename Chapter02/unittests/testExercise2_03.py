import unittest
import os
import json
import pandas as pd

class TestingExercise2_03(unittest.TestCase):
    def setUp(self) -> None:
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

        with open(os.path.join(ROOT_DIR, '..', 'dtypes.json'), 'r') as jsonfile:
            self.dtyp = json.load(jsonfile)
        self.data = pd.read_csv(os.path.join(ROOT_DIR, '..', 'Datasets', 'earthquake_data.csv'),
                                dtype = self.dtyp)

    def test_zeros(self):
        self.time_features = ['month', 'day', 'hour', 'minute', 'second']
        self.data[self.time_features] = self.data[self.time_features].fillna(0)
        self.assertEqual(self.data[self.time_features].describe().T.loc['minute', '50%'], (12.0))

if __name__ == '__main__':
    unittest.main()