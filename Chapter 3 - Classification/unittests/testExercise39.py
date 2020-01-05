import unittest
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.model_selection import train_test_split
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

df = pd.read_csv(os.path.join(ROOT_DIR, '..', 'breast-cancer-data.csv'))

df_test = df.iloc[430]
df = df.drop([430]) # Remove the sample

train_X, valid_X, train_y, valid_y = train_test_split(df[['mean radius', 'worst radius']], df.diagnosis,
                                                      test_size=0.2, random_state=123)

model = KNN(n_neighbors=3)
model.fit(X=train_X, y=train_y)


class TestingExercise39(unittest.TestCase):
    def test_dataframe_shape(self):
        expected_shape = (568, 31)
        actual_shape = df.shape
        self.assertEqual(actual_shape, expected_shape)

    def test_validation_accuracy(self):
        actual_accuracy = model.score(X=valid_X, y=valid_y)
        self.assertAlmostEqual(actual_accuracy, 0.93859649)

    def test_test_smaple_prediction(self):
        pred = model.predict(df_test[['mean radius', 'worst radius']].values.reshape((-1, 2)))[0]
        truth = df_test.diagnosis
        self.assertEqual(pred, 'benign')
        self.assertEqual(truth, 'malignant')


if __name__ == '__main__':
    unittest.main()
