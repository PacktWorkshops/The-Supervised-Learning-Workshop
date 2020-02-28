import unittest
import os
import json
import pandas as pd

class TestingExercise2_17(unittest.TestCase):
    def setUp(self) -> None:
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

        with open(os.path.join(ROOT_DIR, '..', 'dtypes.json'), 'r') as jsonfile:
            self.dtyp = json.load(jsonfile)
        self.data = pd.read_csv(os.path.join(ROOT_DIR, '..', 'Datasets', 'earthquake_data.csv'),
                                dtype = self.dtyp)

    def test_grouping(self):
        self.grouped = self.data.groupby(['intensity', 'flag_tsunami']).size().unstack()
        self.assertEqual(max(self.grouped.iloc[:, 0]), (494))
        self.assertEqual(max(self.grouped.iloc[:, 1].dropna()), (132))


if __name__ == '__main__':
    unittest.main()