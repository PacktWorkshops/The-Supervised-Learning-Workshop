import unittest
import struct
import numpy as np
import gzip
from array import array
from sklearn.linear_model import LinearRegression
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

with gzip.open(os.path.join(ROOT_DIR, '..', 'train-images-idx3-ubyte.gz'), 'rb') as f:
    magic, size, rows, cols = struct.unpack(">IIII", f.read(16))
    img = np.array(array("B", f.read())).reshape((size, rows, cols))

with gzip.open(os.path.join(ROOT_DIR, '..', 'train-labels-idx1-ubyte.gz'), 'rb') as f:
    magic, size = struct.unpack(">II", f.read(8))
    labels = np.array(array("B", f.read()))

with gzip.open(os.path.join(ROOT_DIR, '..', 't10k-images-idx3-ubyte.gz'), 'rb') as f:
    magic, size, rows, cols = struct.unpack(">IIII", f.read(16))

    img_test = np.array(array("B", f.read())).reshape((size, rows, cols))

with gzip.open(os.path.join(ROOT_DIR, '..', 't10k-labels-idx1-ubyte.gz'), 'rb') as f:
    magic, size = struct.unpack(">II", f.read(8))
    labels_test = np.array(array("B", f.read()))

samples_0_1 = np.where((labels == 0) | (labels == 1))[0]
images_0_1 = img[samples_0_1].reshape((-1, rows * cols))
labels_0_1 = labels[samples_0_1]

model = LinearRegression()
model.fit(X=images_0_1, y=labels_0_1)


class TestingActivity11(unittest.TestCase):
    def test_dimensions(self):
        self.assertEqual(images_0_1.shape, (12665, 784))
        self.assertEqual(labels_0_1.shape, (12665,))

    def test_r_square_score(self):
        train_r_square_score = model.score(X=images_0_1, y=labels_0_1)
        self.assertAlmostEqual(train_r_square_score, 0.97057898)

    def test_train_accuracy(self):
        y_pred = model.predict(images_0_1) > 0.5
        y_pred = y_pred.astype(int)
        train_accuracy = np.sum(y_pred == labels_0_1) / len(labels_0_1)
        self.assertAlmostEqual(train_accuracy, 0.99478879)

    def test_test_accuracy(self):
        samples_0_1_test = np.where((labels_test == 0) | (labels_test == 1))
        images_0_1_test = img_test[samples_0_1_test].reshape((-1, rows * cols))
        labels_0_1_test = labels_test[samples_0_1_test]
        y_pred = model.predict(images_0_1_test) > 0.5
        y_pred = y_pred.astype(int)
        test_accuracy = np.sum(y_pred == labels_0_1_test) / len(labels_0_1_test)
        self.assertAlmostEqual(test_accuracy, 0.99243499)


if __name__ == '__main__':
    unittest.main()