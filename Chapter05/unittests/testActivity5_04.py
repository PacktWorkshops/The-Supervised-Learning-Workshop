import unittest
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import os


class TestingActivity5_04(unittest.TestCase):
    def setUp(self) -> None:
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

        self.df = pd.read_csv(os.path.join(ROOT_DIR, '..', 'Datasets', 'breast-cancer-data.csv'))

        self.X, self.y = self.df[[c for c in self.df.columns if c != 'diagnosis']], self.df.diagnosis

        X_array = self.X.values  # returns a numpy array
        min_max_scaler = preprocessing.MinMaxScaler()
        X_array_scaled = min_max_scaler.fit_transform(X_array)
        self.X = pd.DataFrame(X_array_scaled, columns=self.X.columns)

        diagnoses = [
            'benign',  # 0
            'malignant',  # 1
        ]
        output = [diagnoses.index(diag) for diag in self.y]

        self.train_X, self.valid_X, self.train_y, self.valid_y = train_test_split(self.X, output,
                                                              test_size=0.2, random_state=123)

        self.model = MLPClassifier(solver='sgd', hidden_layer_sizes=(100,), max_iter=1000, random_state=1,
                              learning_rate_init=.01)
        self.model.fit(X=self.train_X, y=self.train_y)

    def test_shape(self):
        self.assertEqual(self.X.shape, (569, 30))

    def test_validation_accuracy(self):
        valid_accuracy = self.model.score(self.valid_X, self.valid_y)
        self.assertAlmostEqual(valid_accuracy, 0.98245614, places=4)


if __name__ == '__main__':
    unittest.main()
