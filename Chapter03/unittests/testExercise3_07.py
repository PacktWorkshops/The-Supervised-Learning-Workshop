import unittest
import os
import pandas as pd
from sklearn.linear_model import LinearRegression

class TestingExercise3_07(unittest.TestCase):
    def setUp(self) -> None:
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

        self.data = pd.read_csv(os.path.join(ROOT_DIR, '..', 'Datasets', 'combined_cycle_power_plant.csv'))

    def test_MLR(self):
        self.X_train = self.data.drop('PE', axis = 1)
        self.Y_train = self.data['PE']
        self.model = LinearRegression()
        self.model.fit(self.X_train, self.Y_train)
        self.Y_pred = self.model.predict(self.X_train)
        self.r2 = self.model.score(self.X_train, self.Y_train)
        self.assertEqual(round(self.r2, 3), (0.929))

if __name__ == '__main__':
    unittest.main()