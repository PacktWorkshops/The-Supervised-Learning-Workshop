import unittest
import os
import pandas as pd

class TestingActivity1_01(unittest.TestCase):
    def setUp(self) -> None:
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

        self.data = pd.read_csv(os.path.join(ROOT_DIR, '..', 'Datasets', 'titanic.csv'))

    def test_head(self):
        self.record = self.data.head(10)
        self.assertEqual(self.record.iloc[0, 0], (0))
        self.assertEqual(self.record.iloc[9, 0], (9))
        self.assertEqual(self.record.Ticket.values[0], ('A/5 21171'))
        self.assertEqual(self.record.Ticket.values[9], (str(237736)))


if __name__ == '__main__':
    unittest.main()