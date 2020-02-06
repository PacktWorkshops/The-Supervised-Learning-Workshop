import unittest
import pandas as pd
import numpy as np
import os


class TestingExercise5_06(unittest.TestCase):
    def setUp(self) -> None:
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

        self.df = pd.DataFrame()
        self.df['Outlook'] = [
            'sunny', 'sunny', 'overcast', 'rain', 'rain', 'rain',
            'overcast', 'sunny', 'sunny', 'rain', 'sunny',
            'overcast', 'overcast', 'rain'
        ]
        self.df['Temperature'] = [
            'hot', 'hot', 'hot', 'mild', 'cool', 'cool', 'cool',
            'mild', 'cool', 'mild', 'mild', 'mild', 'hot', 'mild',
        ]
        self.df['Humidity'] = [
            'high', 'high', 'high', 'high', 'normal', 'normal', 'normal',
            'high', 'normal', 'normal', 'normal', 'high', 'normal', 'high'
        ]
        self.df['Windy'] = [
            'Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Weak', 'Weak',
            'Strong', 'Strong', 'Weak', 'Strong'
        ]
        self.df['Decision'] = [
            'N', 'N', 'P', 'P', 'P', 'N', 'P', 'N', 'P', 'P',
            'P', 'P', 'P', 'N'
        ]

        # Probability of P
        p_p = len(self.df.loc[self.df.Decision == 'P']) / len(self.df)
        # Probability of N
        p_n = len(self.df.loc[self.df.Decision == 'N']) / len(self.df)
        self.entropy_decision = -p_n * np.log2(p_n) - p_p * np.log2(p_p)

    def IG(self, data, column, ent_decision):
        IG_decision = ent_decision
        for name, temp in data.groupby(column):
            p_p = len(temp.loc[temp.Decision == 'P']) / len(temp)
            p_n = len(temp.loc[temp.Decision != 'P']) / len(temp)
            entropy_decision = 0
            if p_p != 0:
                entropy_decision -= (p_p) * np.log2(p_p)
            if p_n != 0:
                entropy_decision -= (p_n) * np.log2(p_n)
            IG_decision -= (len(temp) / len(self.df)) * entropy_decision
        return IG_decision

    @staticmethod
    def f_entropy_decision(data):
        p_p = len(data.loc[data.Decision == 'P']) / len(data)
        p_n = len(data.loc[data.Decision == 'N']) / len(data)
        return -p_n * np.log2(p_n) - p_p * np.log2(p_p)

    def test_dataframe_shape(self):
        expected_shape = (14, 5)
        actual_shape = self.df.shape
        self.assertEqual(actual_shape, expected_shape)

    def test_entropy(self):
        self.assertAlmostEqual(self.entropy_decision, 0.9402860, places=4)

    def test_outlook_info_gain(self):
        IG_decision_Outlook = self.entropy_decision  # H(S)
        # Iterate through the values for outlook and compute the probabilities
        # and entropy values
        for name, Outlook in self.df.groupby('Outlook'):
            num_p = len(Outlook.loc[Outlook.Decision == 'P'])
            num_n = len(Outlook.loc[Outlook.Decision != 'P'])
            num_Outlook = len(Outlook)
            entropy_decision_outlook = 0
            # Cannot compute log of 0 so add checks
            if num_p != 0:
                entropy_decision_outlook -= (num_p / num_Outlook) \
                                            * np.log2(num_p / num_Outlook)
            # Cannot compute log of 0 so add checks
            if num_n != 0:
                entropy_decision_outlook -= (num_n / num_Outlook) \
                                            * np.log2(num_n / num_Outlook)
            IG_decision_Outlook -= (num_Outlook / len(self.df)) * entropy_decision_outlook
        self.assertAlmostEqual(IG_decision_Outlook, 0.24674982, places=4)

    def test_individual_outlook_info_gains(self):
        self.assertAlmostEqual(self.IG(self.df, 'Outlook', self.entropy_decision), 0.24674982, places=4)
        self.assertAlmostEqual(self.IG(self.df, 'Temperature', self.entropy_decision), 0.02922256, places=4)
        self.assertAlmostEqual(self.IG(self.df, 'Humidity', self.entropy_decision), 0.15183550, places=4)
        self.assertAlmostEqual(self.IG(self.df, 'Windy', self.entropy_decision), 0.04812703, places=4)

    def test_sunny_outlook_entropy(self):
        df_next = self.df.loc[self.df.Outlook != 'overcast']
        df_sunny = df_next.loc[df_next.Outlook == 'sunny']
        entropy_decision = self.f_entropy_decision(df_sunny)
        self.assertAlmostEqual(entropy_decision, 0.97095059, places=4)

    def test_sunny_outlook_info_gain(self):
        df_next = self.df.loc[self.df.Outlook != 'overcast']
        df_sunny = df_next.loc[df_next.Outlook == 'sunny']
        entropy_decision = self.f_entropy_decision(df_sunny)
        self.assertAlmostEqual(self.IG(df_sunny, 'Temperature', entropy_decision), 0.82809345, places=4)
        self.assertAlmostEqual(self.IG(df_sunny, 'Humidity', entropy_decision), 0.97095059, places=4)
        self.assertAlmostEqual(self.IG(df_sunny, 'Windy', entropy_decision), 0.63131577, places=4)


if __name__ == '__main__':
    unittest.main()
