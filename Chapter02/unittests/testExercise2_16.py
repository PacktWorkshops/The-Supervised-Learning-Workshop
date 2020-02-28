import unittest
import os
import json
import pandas as pd

class TestingExercise2_16(unittest.TestCase):
    def setUp(self) -> None:
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

        with open(os.path.join(ROOT_DIR, '..', 'dtypes.json'), 'r') as jsonfile:
            self.dtyp = json.load(jsonfile)
        self.data = pd.read_csv(os.path.join(ROOT_DIR, '..', 'Datasets', 'earthquake_data.csv'),
                                dtype = self.dtyp)

    def test_plot_data(self):
        self.data.loc[:,'flag_tsunami'] = self.data.flag_tsunami.apply(lambda t: int(str(t) == 'Tsu'))
        self.country_counts = self.data.country.value_counts()
        self.top_countries = self.country_counts[self.country_counts > 100]
        self.assertEqual(len(self.top_countries), (13))
        self.assertEqual(max(self.top_countries), (590))


if __name__ == '__main__':
    unittest.main()