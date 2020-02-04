import unittest
import struct
import numpy as np
import gzip
from array import array
from sklearn.neural_network import MLPClassifier
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

with gzip.open(os.path.join(ROOT_DIR, '..', 'Datasets', 'train-images-idx3-ubyte.gz'), 'rb') as f:
    magic, size, rows, cols = struct.unpack(">IIII", f.read(16))
    img = np.array(array("B", f.read())).reshape((size, rows, cols))

with gzip.open(os.path.join(ROOT_DIR, '..', 'Datasets', 'train-labels-idx1-ubyte.gz'), 'rb') as f:
    magic, size = struct.unpack(">II", f.read(8))
    labels = np.array(array("B", f.read()))

with gzip.open(os.path.join(ROOT_DIR, '..', 'Datasets', 't10k-images-idx3-ubyte.gz'), 'rb') as f:
    magic, size, rows, cols = struct.unpack(">IIII", f.read(16))

    img_test = np.array(array("B", f.read())).reshape((size, rows, cols))

with gzip.open(os.path.join(ROOT_DIR, '..', 'Datasets', 't10k-labels-idx1-ubyte.gz'), 'rb') as f:
    magic, size = struct.unpack(">II", f.read(8))
    labels_test = np.array(array("B", f.read()))

np.random.seed(0) # Give consistent random numbers
selection = np.random.choice(len(img), 5000)
selected_images = img[selection].reshape((-1, rows * cols))
selected_labels = labels[selection]

selected_images = selected_images / 255.0
img_test = img_test / 255.0

model = MLPClassifier(solver='sgd', hidden_layer_sizes=(100,), max_iter=1000, random_state=1,
                      learning_rate_init=.01)
model.fit(X=selected_images, y=selected_labels)


class TestingExercise43(unittest.TestCase):
    def test_dimensions(self):
        self.assertEqual(selected_images.shape, (5000, 784))
        self.assertEqual(selected_labels.shape, (5000,))

    def test_ann_train_accuracy(self):
        train_accuracy = model.score(X=selected_images, y=selected_labels)
        self.assertEqual(train_accuracy, 1)

    def test_ann_test_accuracy(self):
        test_accuracy = model.score(X=img_test.reshape((-1, rows * cols)), y=labels_test)
        self.assertAlmostEqual(test_accuracy, 0.9384, places=4)

    def test_predictions(self):
        pred_label = model.predict(selected_images)[0]
        pred_proba = model.predict_proba(selected_images)[0][4]
        self.assertEqual(pred_label, 4)
        self.assertAlmostEqual(pred_proba, 0.99999524, places=4)


if __name__ == '__main__':
    unittest.main()
