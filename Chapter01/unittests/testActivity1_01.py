import unittest
import os
import pandas as pd

class TestingActivity1_01(unittest.TestCase):
    def setUp(self) -> None:
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

        self.data = pd.read_csv(os.path.join(ROOT_DIR, '..', 'Datasets', 'titanic.csv'))

    def test_record(self):
        self.record = self.data.loc[(self.data.Fare.isna())]
        self.assertEqual(self.record.iloc[0, 0], (1043))
        self.assertEqual(self.record.Ticket.values[0], (str(3701)))


if __name__ == '__main__':
    unittest.main()