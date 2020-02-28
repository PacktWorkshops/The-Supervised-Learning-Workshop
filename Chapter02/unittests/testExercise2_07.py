import unittest
import os
import json
import pandas as pd
import numpy as np

class TestingExercise2_07(unittest.TestCase):
    def setUp(self) -> None:
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

        with open(os.path.join(ROOT_DIR, '..', 'dtypes.json'), 'r') as jsonfile:
            self.dtyp = json.load(jsonfile)
        self.data = pd.read_csv(os.path.join(ROOT_DIR, '..', 'Datasets', 'earthquake_data.csv'),
                                dtype = self.dtyp)

    def test_object_vars(self):
        self.object_variables = self.data.select_dtypes(include = [np.object]).nunique().sort_values()
        self.assertEqual(max(self.object_variables), (3821))

if __name__ == '__main__':
    unittest.main()