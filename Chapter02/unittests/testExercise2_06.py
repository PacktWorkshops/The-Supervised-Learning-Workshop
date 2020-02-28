import unittest
import os
import json
import pandas as pd
import numpy as np

class TestingExercise2_06(unittest.TestCase):
    def setUp(self) -> None:
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

        with open(os.path.join(ROOT_DIR, '..', 'dtypes.json'), 'r') as jsonfile:
            self.dtyp = json.load(jsonfile)
        self.data = pd.read_csv(os.path.join(ROOT_DIR, '..', 'Datasets', 'earthquake_data.csv'),
                                dtype = self.dtyp)

    def test_replace(self):
        self.category_means = self.data[['damage_description', 'damage_millions_dollars']].groupby('damage_description').mean()
        self.replacement_values = self.category_means.damage_millions_dollars.to_dict()
        self.replacement_values['NA'] = -1
        self.replacement_values['0'] = 0
        self.imputed_values = self.data.damage_description.map(self.replacement_values)
        self.data['damage_millions_dollars'] = np.where(self.data.damage_millions_dollars.isnull(),
                                                        self.data.damage_description.map(self.replacement_values),
                                                        self.data.damage_millions_dollars)
        self.assertEqual(self.data.flag_tsunami.value_counts()['No'], (4270))

if __name__ == '__main__':
    unittest.main()