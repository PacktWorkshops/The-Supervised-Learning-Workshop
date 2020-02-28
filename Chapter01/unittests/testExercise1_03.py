import unittest
import os
import pandas as pd

class TestingExercise1_03(unittest.TestCase):
    def setUp(self) -> None:
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

        self.data = pd.read_csv(os.path.join(ROOT_DIR, '..', 'Datasets', 'titanic.csv'))

    def test_slices(self):
        self.records = self.data[self.data.Age < 21][['Name', 'Age']]
        self.assertEqual(self.records.head().reset_index(drop = True).loc[0, 'Age'], (2.0))
        self.assertEqual(len(self.records), (249))
        self.assertEqual(self.data.loc[(self.data.Pclass == 3) |
                                       (self.data.Pclass == 1)].reset_index(drop = True).loc[0, 'Ticket'],
                         ('A/5 21171'))

if __name__ == '__main__':
    unittest.main()