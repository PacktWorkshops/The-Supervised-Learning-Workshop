import unittest
import os
import pandas as pd

class TestingExercise3_01(unittest.TestCase):
    def setUp(self) -> None:
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

        self.data = pd.read_csv(os.path.join(ROOT_DIR, '..', 'Datasets', 'synth_temp.csv'))

    def test_rolling(self):
        self.data = self.data[self.data['Year'] > 1901].groupby('Year').agg('mean').rolling(window = 10).mean().tail(20)
        self.value1 = round(self.data.reset_index(drop = True).iloc[0, 0], 2)
        self.value2 = round(self.data.reset_index(drop = True).iloc[19, 0], 2)
        self.assertEqual(self.value1, (18.91))
        self.assertEqual(self.value2, (19.65))

if __name__ == '__main__':
    unittest.main()