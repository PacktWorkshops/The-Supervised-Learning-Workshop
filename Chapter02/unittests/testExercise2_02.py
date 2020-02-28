import unittest
import os
import json
import pandas as pd

class TestingExercise2_02(unittest.TestCase):
    def setUp(self) -> None:
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

        with open(os.path.join(ROOT_DIR, '..', 'dtypes.json'), 'r') as jsonfile:
            self.dtyp = json.load(jsonfile)
        self.data = pd.read_csv(os.path.join(ROOT_DIR, '..', 'Datasets', 'earthquake_data.csv'),
                                dtype = self.dtyp)

    def test_missing(self):
        self.assertEqual(self.data.isnull().mean().loc['id'], (0))
        self.assertEqual(round(self.data.isnull().mean().loc['total_damage_millions_dollars'], 2), (0.93))

if __name__ == '__main__':
    unittest.main()