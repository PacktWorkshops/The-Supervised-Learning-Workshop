import unittest
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier as KNN
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

df = pd.read_csv(os.path.join(ROOT_DIR, '..', 'breast-cancer-data.csv'))

labelled_diagnoses = [
    'benign',
    'malignant',
]

for idx, label in enumerate(labelled_diagnoses):
    df.diagnosis = df.diagnosis.replace(label, idx)

model = KNN(n_neighbors=3)
model.fit(X=df[['mean radius', 'worst radius']], y=df.diagnosis)


class TestingExercise40(unittest.TestCase):
    def test_dataframe_shape(self):
        expected_shape = (569, 31)
        actual_shape = df.shape
        self.assertEqual(actual_shape, expected_shape)

    def test_meshgrid_values(self):
        spacing = 0.1
        mean_radius_range = np.arange(df['mean radius'].min() - 1, df['mean radius'].max() + 1, spacing)
        worst_radius_range = np.arange(df['worst radius'].min() - 1, df['worst radius'].max() + 1, spacing)
        xx, yy = np.meshgrid(mean_radius_range, worst_radius_range)  # Create the mesh
        pred_x = np.c_[xx.ravel(), yy.ravel()]
        pred_y = model.predict(pred_x).reshape(xx.shape)
        self.assertAlmostEqual(xx[0][0], 5.981)
        self.assertAlmostEqual(yy[0][0], 6.93)
        self.assertAlmostEqual(pred_x[0][1], 6.93)
        self.assertEqual(pred_y[0][0], 0)


if __name__ == '__main__':
    unittest.main()
