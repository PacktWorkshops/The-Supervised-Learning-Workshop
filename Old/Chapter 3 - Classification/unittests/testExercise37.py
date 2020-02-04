import unittest
import struct
import numpy as np
import gzip
from array import array
from sklearn.linear_model import LogisticRegression
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

model = LogisticRegression(solver='liblinear')
model.fit(X=images_0_1, y=labels_0_1)


class TestingExercise37(unittest.TestCase):
    def test_dimensions(self):
        self.assertEqual(images_0_1.shape, (12665, 784))
        self.assertEqual(labels_0_1.shape, (12665,))

    def test_logreg_train_accuracy(self):
        train_accuracy = model.score(X=images_0_1, y=labels_0_1)
        self.assertEqual(train_accuracy, 1)

    def test_logreg_test_accuracy(self):
        samples_0_1_test = np.where((labels_test == 0) | (labels_test == 1))
        images_0_1_test = img_test[samples_0_1_test].reshape((-1, rows * cols))
        labels_0_1_test = labels_test[samples_0_1_test]
        test_accuracy = model.score(X=images_0_1_test, y=labels_0_1_test)
        self.assertAlmostEqual(test_accuracy, 0.99952719)

    def test_predictions(self):
        pred_label = model.predict(images_0_1)[0]
        pred_proba = model.predict_proba(images_0_1)[0][0]
        self.assertEqual(pred_label, 0)
        self.assertAlmostEqual(pred_proba, 0.99999999)


if __name__ == '__main__':
    unittest.main()