import unittest
import os
import pandas as pd
import numpy as np

class TestingExercise1_05(unittest.TestCase):
    def setUp(self) -> None:
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

        self.data = pd.read_csv(os.path.join(ROOT_DIR, '..', 'Datasets', 'titanic.csv'))

    def test_lambda(self):
        self.records = self.data.groupby('Embarked')
        self.assertEqual(self.records.agg(lambda x: x.values[0]).loc['C', 'Cabin'], ('C85'))
        self.assertEqual(round(self.records.agg({'Fare' : np.sum,
                                                 'Age' : lambda x: x.values[0]}).loc['S', 'Fare'], 1), (25033.4))


if __name__ == '__main__':
    unittest.main()