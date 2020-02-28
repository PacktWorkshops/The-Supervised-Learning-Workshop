import unittest
import os
import pandas as pd

class TestingActivity2_01(unittest.TestCase):
    def setUp(self) -> None:
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

        self.data = pd.read_csv(os.path.join(ROOT_DIR, '..', 'Datasets', 'house_prices.csv'))

    def test_missing(self):
        self.describe = self.data.describe().T
        self.assertEqual(self.describe.iloc[0, 0], (1460))
        self.missing_data = self.data.isnull().mean()*100
        self.missing_data = self.missing_data[self.missing_data > 0]
        self.assertEqual(round(self.missing_data.loc['LotFrontage'], 4), (17.7397))
        self.assertEqual(self.data['FireplaceQu'].fillna('NA')[0], ('NA'))

if __name__ == '__main__':
    unittest.main()