import unittest
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.model_selection import train_test_split
import os


class TestingExercise5_05(unittest.TestCase):
    def setUp(self) -> None:
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

        self.df = pd.read_csv(os.path.join(ROOT_DIR, '..', 'Datasets', 'breast-cancer-data.csv'))

        labelled_diagnoses = [
            'benign',
            'malignant',
        ]

        for idx, label in enumerate(labelled_diagnoses):
            self.df.diagnosis = self.df.diagnosis.replace(label, idx)

        self.train_X, self.valid_X, self.train_y, self.valid_y = train_test_split(self.df[['mean radius', 'worst radius']],
                                                              self.df.diagnosis,
                                                              test_size=0.2, random_state=123)

        self.model = KNN(n_neighbors=3)
        self.model.fit(X=self.train_X[['mean radius', 'worst radius']], y=self.train_y)

    def test_dataframe_shape(self):
        expected_shape = (569, 31)
        actual_shape = self.df.shape
        self.assertEqual(actual_shape, expected_shape)

    def test_meshgrid_values(self):
        spacing = 0.1
        mean_radius_range = np.arange(self.df['mean radius'].min() - 1, self.df['mean radius'].max() + 1, spacing)
        worst_radius_range = np.arange(self.df['worst radius'].min() - 1, self.df['worst radius'].max() + 1, spacing)
        xx, yy = np.meshgrid(mean_radius_range, worst_radius_range)  # Create the mesh
        self.assertAlmostEqual(xx[0][0], 5.981, places=2)
        self.assertAlmostEqual(yy[0][0], 6.93, places=2)

    def test_predictions(self):
        spacing = 0.1
        mean_radius_range = np.arange(self.df['mean radius'].min() - 1, self.df['mean radius'].max() + 1, spacing)
        worst_radius_range = np.arange(self.df['worst radius'].min() - 1, self.df['worst radius'].max() + 1, spacing)
        xx, yy = np.meshgrid(mean_radius_range, worst_radius_range)  # Create the mesh
        pred_x = np.c_[xx.ravel(), yy.ravel()]
        pred_y = self.model.predict(pred_x).reshape(xx.shape)
        self.assertAlmostEqual(pred_x[0][1], 6.93, places=2)
        self.assertEqual(pred_y[0][0], 0)

    def test_validation_accuracy(self):
        self.assertAlmostEqual(self.model.score(X=self.valid_X, y=self.valid_y), 0.921052632, places=2)

    def test_train_accuracy(self):
        self.assertAlmostEqual(self.model.score(X=self.train_X, y=self.train_y), 0.934065934, places=2)


if __name__ == '__main__':
    unittest.main()
