import unittest
import os
import json
import pandas as pd

class TestingExercise2_01(unittest.TestCase):
    def setUp(self) -> None:
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

        with open(os.path.join(ROOT_DIR, '..', 'dtypes.json'), 'r') as jsonfile:
            self.dtyp = json.load(jsonfile)
        self.data = pd.read_csv(os.path.join(ROOT_DIR, '..', 'Datasets', 'earthquake_data.csv'),
                                dtype = self.dtyp)

    def test_data(self):
        self.assertEqual(self.dtyp['id'], ('float'))
        self.assertEqual(self.data.dtypes['month'], ('float64'))
        self.assertEqual(round(self.data.head(5).iloc[0, 0], 0), (338))
        self.assertEqual(round(self.data.describe().T.loc['total_damage_millions_dollars', 'max'], 2), (220085.46))

if __name__ == '__main__':
    unittest.main()