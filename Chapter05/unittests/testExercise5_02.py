import unittest
import struct
import numpy as np
import gzip
from array import array
from sklearn.linear_model import LogisticRegression
import os


class TestingExercise5_02(unittest.TestCase):
    def setUp(self) -> None:
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

        with gzip.open(os.path.join(ROOT_DIR, '..', 'Datasets', 'train-images-idx3-ubyte.gz'), 'rb') as f:
            magic, size, rows, cols = struct.unpack(">IIII", f.read(16))
            img = np.array(array("B", f.read())).reshape((size, rows, cols))

        with gzip.open(os.path.join(ROOT_DIR, '..', 'Datasets', 'train-labels-idx1-ubyte.gz'), 'rb') as f:
            magic, size = struct.unpack(">II", f.read(8))
            labels = np.array(array("B", f.read()))

        with gzip.open(os.path.join(ROOT_DIR, '..', 'Datasets', 't10k-images-idx3-ubyte.gz'), 'rb') as f:
            magic, size, self.rows, self.cols = struct.unpack(">IIII", f.read(16))

            self.img_test = np.array(array("B", f.read())).reshape((size, rows, cols))

        with gzip.open(os.path.join(ROOT_DIR, '..', 'Datasets', 't10k-labels-idx1-ubyte.gz'), 'rb') as f:
            magic, size = struct.unpack(">II", f.read(8))
            self.labels_test = np.array(array("B", f.read()))

        samples_0_1 = np.where((labels == 0) | (labels == 1))[0]
        self.images_0_1 = img[samples_0_1].reshape((-1, rows * cols))
        self.labels_0_1 = labels[samples_0_1]

        self.model = LogisticRegression(solver='liblinear')
        self.model.fit(X=self.images_0_1, y=self.labels_0_1)

    def test_dimensions(self):
        self.assertEqual(self.images_0_1.shape, (12665, 784))
        self.assertEqual(self.labels_0_1.shape, (12665,))

    def test_logreg_train_accuracy(self):
        train_accuracy = self.model.score(X=self.images_0_1, y=self.labels_0_1)
        self.assertEqual(train_accuracy, 1)

    def test_logreg_test_accuracy(self):
        samples_0_1_test = np.where((self.labels_test == 0) | (self.labels_test == 1))
        images_0_1_test = self.img_test[samples_0_1_test].reshape((-1, self.rows * self.cols))
        labels_0_1_test = self.labels_test[samples_0_1_test]
        test_accuracy = self.model.score(X=images_0_1_test, y=labels_0_1_test)
        self.assertAlmostEqual(test_accuracy, 0.99952719, places=4)

    def test_predictions(self):
        pred_label = self.model.predict(self.images_0_1)[0]
        pred_proba = self.model.predict_proba(self.images_0_1)[0][0]
        self.assertEqual(pred_label, 0)
        self.assertAlmostEqual(pred_proba, 0.99999999, places=4)


if __name__ == '__main__':
    unittest.main()