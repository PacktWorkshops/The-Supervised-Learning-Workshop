import unittest
import os
import pandas as pd
import numpy as np

class TestingActivity3_04(unittest.TestCase):
    def setUp(self) -> None:
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

        self.data = pd.read_csv(os.path.join(ROOT_DIR, '..', 'Datasets', 'austin_weather.csv'))

    def test_sin_cos(self):
        self.data.loc[:, 'Year'] = self.data.loc[:, 'Date'].str.slice(0, 4).astype('int')
        self.data_2015 = self.data[self.data['Year'].eq(2015)].reset_index(drop = True)
        self.data_2015['Day_of_Year'] = self.data_2015.index + 1
        self.data_2015['sine_Day'] = np.sin( 2 * np.pi * self.data_2015['Day_of_Year'] / 365)
        self.data_2015['cosine_Day'] = np.cos(2 * np.pi * self.data_2015['Day_of_Year'] / 365)
        self.assertEqual(round(self.data_2015.loc[364, 'sine_Day'], 3), (0))
        self.assertEqual(round(self.data_2015.loc[364, 'cosine_Day'], 3), (1))

if __name__ == '__main__':
    unittest.main()