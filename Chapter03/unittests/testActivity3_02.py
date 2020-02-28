import unittest
import os
import pandas as pd

class TestingActivity3_02(unittest.TestCase):
    def setUp(self) -> None:
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

        self.data = pd.read_csv(os.path.join(ROOT_DIR, '..', 'Datasets', 'austin_weather.csv'))

    def test_subset(self):
        self.data.loc[:, 'Year'] = self.data.loc[:, 'Date'].str.slice(0, 4).astype('int')
        self.data_2015 = self.data[self.data['Year'].eq(2015)].reset_index(drop = True)
        self.assertEqual(self.data_2015.loc[364, 'Date'], ('2015-12-31'))


if __name__ == '__main__':
    unittest.main()