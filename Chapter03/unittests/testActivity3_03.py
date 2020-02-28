import unittest
import os
import pandas as pd

class TestingActivity3_03(unittest.TestCase):
    def setUp(self) -> None:
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

        self.data = pd.read_csv(os.path.join(ROOT_DIR, '..', 'Datasets', 'austin_weather.csv'))

    def test_dummies(self):
        self.data.loc[:, 'Year'] = self.data.loc[:, 'Date'].str.slice(0, 4).astype('int')
        self.data_2015 = self.data[self.data['Year'].eq(2015)].reset_index(drop = True)
        self.data_2015.loc[:, 'Month'] = self.data_2015.loc[:, 'Date'].str.slice(5, 7).astype('int')
        self.dummy_vars = pd.get_dummies(self.data_2015['Month'], drop_first = True)
        self.dummy_vars.columns = ['Feb', 'Mar', 'Apr', 'May', 'Jun',
                                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        self.data_2015 = pd.concat([self.data_2015, self.dummy_vars],
                                   axis = 1)
        self.assertEqual(self.data_2015.loc[364, 'Dec'], (1))

if __name__ == '__main__':
    unittest.main()