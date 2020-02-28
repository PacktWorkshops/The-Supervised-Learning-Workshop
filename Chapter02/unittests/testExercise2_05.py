import unittest
import os
import json
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

class TestingExercise2_05(unittest.TestCase):
    def setUp(self) -> None:
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

        with open(os.path.join(ROOT_DIR, '..', 'dtypes.json'), 'r') as jsonfile:
            self.dtyp = json.load(jsonfile)
        self.data = pd.read_csv(os.path.join(ROOT_DIR, '..', 'Datasets', 'earthquake_data.csv'),
                                dtype = self.dtyp)
        self.imp = SimpleImputer(missing_values = np.nan, strategy = 'constant', fill_value = 'NA')

    def test_impute(self):
        self.description_features = ['injuries_description', 'damage_description', 'total_injuries_description', 'total_damage_description']
        self.assertEqual(self.data[pd.isnull(self.data.damage_millions_dollars)].shape[0], (5594))

if __name__ == '__main__':
    unittest.main()