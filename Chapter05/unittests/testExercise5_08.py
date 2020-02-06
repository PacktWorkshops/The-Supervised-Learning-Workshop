import unittest
import struct
import numpy as np
import gzip
from array import array
from sklearn.neural_network import MLPClassifier
import os


class TestingExercise5_08(unittest.TestCase):
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

            img_test = np.array(array("B", f.read())).reshape((size, rows, cols))

        with gzip.open(os.path.join(ROOT_DIR, '..', 'Datasets', 't10k-labels-idx1-ubyte.gz'), 'rb') as f:
            magic, size = struct.unpack(">II", f.read(8))
            self.labels_test = np.array(array("B", f.read()))

        np.random.seed(0)  # Give consistent random numbers
        selection = np.random.choice(len(img), 5000)
        self.selected_images = img[selection].reshape((-1, rows * cols))
        self.selected_labels = labels[selection]

        self.selected_images = self.selected_images / 255.0
        self.img_test = img_test / 255.0

        self.model = MLPClassifier(solver='sgd', hidden_layer_sizes=(100,), max_iter=1000, random_state=1,
                              learning_rate_init=.01)
        self.model.fit(X=self.selected_images, y=self.selected_labels)

    def test_dimensions(self):
        self.assertEqual(self.selected_images.shape, (5000, 784))
        self.assertEqual(self.selected_labels.shape, (5000,))

    def test_ann_train_accuracy(self):
        train_accuracy = self.model.score(X=self.selected_images, y=self.selected_labels)
        self.assertEqual(train_accuracy, 1)

    def test_ann_test_accuracy(self):
        test_accuracy = self.model.score(X=self.img_test.reshape((-1, self.rows * self.cols)), y=self.labels_test)
        self.assertAlmostEqual(test_accuracy, 0.9384, places=4)

    def test_predictions(self):
        pred_label = self.model.predict(self.selected_images)[0]
        pred_proba = self.model.predict_proba(self.selected_images)[0][4]
        self.assertEqual(pred_label, 4)
        self.assertAlmostEqual(pred_proba, 0.99999524, places=4)


if __name__ == '__main__':
    unittest.main()
