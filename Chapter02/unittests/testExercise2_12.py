import unittest
import os
import json
import pandas as pd

class TestingExercise2_12(unittest.TestCase):
    def setUp(self) -> None:
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

        with open(os.path.join(ROOT_DIR, '..', 'dtypes.json'), 'r') as jsonfile:
            self.dtyp = json.load(jsonfile)
        self.data = pd.read_csv(os.path.join(ROOT_DIR, '..', 'Datasets', 'earthquake_data.csv'),
                                dtype = self.dtyp)

    def test_filter(self):
        self.assertEqual(max(self.data[~pd.isnull(self.data.injuries) & ~pd.isnull(self.data.eq_primary)]['injuries']), (799000))

if __name__ == '__main__':
    unittest.main()