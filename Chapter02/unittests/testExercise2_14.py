import unittest
import os
import json
import pandas as pd

class TestingExercise2_14(unittest.TestCase):
    def setUp(self) -> None:
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

        with open(os.path.join(ROOT_DIR, '..', 'dtypes.json'), 'r') as jsonfile:
            self.dtyp = json.load(jsonfile)
        self.data = pd.read_csv(os.path.join(ROOT_DIR, '..', 'Datasets', 'earthquake_data.csv'),
                                dtype = self.dtyp)

    def test_subset_corr(self):
        self.data_subset = self.data[['focal_depth', 'eq_primary', 'eq_mag_mw', 'eq_mag_ms', 'eq_mag_mb', 'intensity']]
        self.assertEqual(round(self.data_subset.corr().loc['eq_primary', 'eq_mag_mb'], 4), (0.8998))


if __name__ == '__main__':
    unittest.main()