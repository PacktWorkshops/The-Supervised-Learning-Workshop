import unittest
import os
import json
import pandas as pd

class TestingExercise2_15(unittest.TestCase):
    def setUp(self) -> None:
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

        with open(os.path.join(ROOT_DIR, '..', 'dtypes.json'), 'r') as jsonfile:
            self.dtyp = json.load(jsonfile)
        self.data = pd.read_csv(os.path.join(ROOT_DIR, '..', 'Datasets', 'earthquake_data.csv'),
                                dtype = self.dtyp)

    def test_plot_data(self):
        self.data.loc[:,'flag_tsunami'] = self.data.flag_tsunami.apply(lambda t: int(str(t) == 'Tsu'))
        self.subset = self.data[~pd.isnull(self.data.intensity)][['intensity','flag_tsunami']]
        self.data_to_plot = self.subset.groupby('intensity').sum()
        self.assertEqual(max(self.data_to_plot.loc[:, 'flag_tsunami']), (132))


if __name__ == '__main__':
    unittest.main()