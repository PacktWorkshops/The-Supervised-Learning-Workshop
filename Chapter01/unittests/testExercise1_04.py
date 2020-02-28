import unittest
import os
import pandas as pd
import numpy as np

class TestingExercise1_04(unittest.TestCase):
    def setUp(self) -> None:
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

        self.data = pd.read_csv(os.path.join(ROOT_DIR, '..', 'Datasets', 'titanic.csv'))

    def test_groups(self):
        self.records = self.data.groupby('Embarked')
        self.assertEqual(len(self.records), (3))
        self.assertEqual(round(self.records.agg(np.mean).loc['C', 'Fare'], 3), (62.336))


if __name__ == '__main__':
    unittest.main()