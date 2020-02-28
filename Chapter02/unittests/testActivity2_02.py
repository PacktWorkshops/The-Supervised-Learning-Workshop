import unittest
import os
import pandas as pd
import numpy as np

class TestingActivity2_02(unittest.TestCase):
    def setUp(self) -> None:
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

        self.data = pd.read_csv(os.path.join(ROOT_DIR, '..', 'Datasets', 'house_prices.csv'))

    def test_vars(self):
        self.types = self.data.select_dtypes(include = [np.object]).nunique().sort_values()
        self.assertEqual(self.types[0], (2))
        self.assertEqual(self.types.loc['GarageType'], (6))
        self.counts1S = self.data.HouseStyle.value_counts(dropna = False).reset_index().sort_values(by = 'index')
        self.counts1S = self.counts1S[self.counts1S['index'].eq('1Story')]
        self.assertEqual(self.counts1S.loc[0, 'HouseStyle'], (726))
        self.numerics = self.data.select_dtypes(include=[np.number]).nunique().sort_values(ascending=False)
        self.assertEqual(self.numerics.loc['LotArea'], (1073))
        self.assertEqual(round(self.data.kurt().loc['LotArea'], 0), (203))

if __name__ == '__main__':
    unittest.main()