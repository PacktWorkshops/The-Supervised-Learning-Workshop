import unittest
import os
import json
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

class TestingExercise2_04(unittest.TestCase):
    def setUp(self) -> None:
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

        with open(os.path.join(ROOT_DIR, '..', 'dtypes.json'), 'r') as jsonfile:
            self.dtyp = json.load(jsonfile)
        self.data = pd.read_csv(os.path.join(ROOT_DIR, '..', 'Datasets', 'earthquake_data.csv'),
                                dtype = self.dtyp)
        self.imp = SimpleImputer(missing_values = np.nan, strategy = 'constant', fill_value = 'NA')

    def test_impute(self):
        self.description_features = ['injuries_description', 'damage_description', 'total_injuries_description', 'total_damage_description']
        self.data[self.description_features] = self.imp.fit_transform(self.data[self.description_features])
        self.assertEqual(self.data[self.description_features].describe().T.loc['damage_description', 'top'], ('NA'))

if __name__ == '__main__':
    unittest.main()