import unittest
import os
import pandas as pd

class TestingExercise1_02(unittest.TestCase):
    def setUp(self) -> None:
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

        self.data = pd.read_csv(os.path.join(ROOT_DIR, '..', 'Datasets', 'titanic.csv'))

    def test_slices(self):
        self.records = self.data.iloc[[0, 1, 2]]
        self.assertEqual(self.records.loc[0, 'Age'], (22.0))
        self.assertEqual(self.records.loc[2, 'Fare'], (7.9250))
        self.assertEqual(self.data.iloc[2]['Fare'], (7.9250))

if __name__ == '__main__':
    unittest.main()