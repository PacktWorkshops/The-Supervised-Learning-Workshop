import unittest
import os
import pandas as pd

class TestingActivity2_03(unittest.TestCase):
    def setUp(self) -> None:
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

        self.data = pd.read_csv(os.path.join(ROOT_DIR, '..', 'Datasets', 'house_prices.csv'))

    def test_corr(self):
        self.assertEqual(round(self.data.corr().loc['MSSubClass', 'SalePrice'], 5), (-0.08428))

if __name__ == '__main__':
    unittest.main()