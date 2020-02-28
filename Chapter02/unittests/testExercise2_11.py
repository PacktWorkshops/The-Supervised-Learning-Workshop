import unittest
import os
import json
import pandas as pd

class TestingExercise2_11(unittest.TestCase):
    def setUp(self) -> None:
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

        with open(os.path.join(ROOT_DIR, '..', 'dtypes.json'), 'r') as jsonfile:
            self.dtyp = json.load(jsonfile)
        self.data = pd.read_csv(os.path.join(ROOT_DIR, '..', 'Datasets', 'earthquake_data.csv'),
                                dtype = self.dtyp)

    def test_stat(self):
        self.assertEqual(round(max(self.data.kurt()), 2), (672.95))

if __name__ == '__main__':
    unittest.main()