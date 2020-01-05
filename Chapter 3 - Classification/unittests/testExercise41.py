import unittest
import pandas as pd
import numpy as np
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

df = pd.DataFrame()
df['Outlook'] = [
    'sunny', 'sunny', 'overcast', 'rain', 'rain', 'rain',
    'overcast', 'sunny', 'sunny', 'rain', 'sunny',
    'overcast', 'overcast', 'rain'
]
df['Temperature'] = [
    'hot', 'hot', 'hot', 'mild', 'cool', 'cool', 'cool',
    'mild', 'cool', 'mild', 'mild', 'mild', 'hot', 'mild',
]
df['Humidity'] = [
    'high', 'high', 'high', 'high', 'normal', 'normal', 'normal',
    'high', 'normal', 'normal', 'normal', 'high', 'normal', 'high'
]
df['Windy'] = [
    'Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Weak', 'Weak',
    'Strong', 'Strong', 'Weak', 'Strong'
]
df['Decision'] = [
    'N', 'N', 'P', 'P', 'P', 'N', 'P', 'N', 'P', 'P',
    'P', 'P', 'P', 'N'
]

# Probability of P
p_p = len(df.loc[df.Decision == 'P']) / len(df)
# Probability of N
p_n = len(df.loc[df.Decision == 'N']) / len(df)
entropy_decision = -p_n * np.log2(p_n) - p_p * np.log2(p_p)


def f_entropy_decision(data):
    p_p = len(data.loc[data.Decision == 'P']) / len(data)
    p_n = len(data.loc[data.Decision == 'N']) / len(data)
    return -p_n * np.log2(p_n) - p_p * np.log2(p_p)


def IG(data, column, ent_decision):
    IG_decision = ent_decision
    for name, temp in data.groupby(column):
        p_p = len(temp.loc[temp.Decision == 'P']) / len(temp)
        p_n = len(temp.loc[temp.Decision != 'P']) / len(temp)
        entropy_decision = 0
        if p_p != 0:
            entropy_decision -= (p_p) * np.log2(p_p)
        if p_n != 0:
            entropy_decision -= (p_n) * np.log2(p_n)
        IG_decision -= (len(temp) / len(df)) * entropy_decision
    return IG_decision


class TestingExercise41(unittest.TestCase):
    def test_dataframe_shape(self):
        expected_shape = (14, 5)
        actual_shape = df.shape
        self.assertEqual(actual_shape, expected_shape)

    def test_entropy(self):
        self.assertAlmostEqual(entropy_decision, 0.9402860)

    def test_outlook_info_gain(self):
        IG_decision_Outlook = entropy_decision  # H(S)
        # Iterate through the values for outlook and compute the probabilities
        # and entropy values
        for name, Outlook in df.groupby('Outlook'):
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
            IG_decision_Outlook -= (num_Outlook / len(df)) * entropy_decision_outlook
        self.assertAlmostEqual(IG_decision_Outlook, 0.24674982)

    def test_individual_outlook_info_gains(self):
        self.assertAlmostEqual(IG(df, 'Outlook', entropy_decision), 0.24674982)
        self.assertAlmostEqual(IG(df, 'Temperature', entropy_decision), 0.02922256)
        self.assertAlmostEqual(IG(df, 'Humidity', entropy_decision), 0.15183550)
        self.assertAlmostEqual(IG(df, 'Windy', entropy_decision), 0.04812703)

    def test_sunny_outlook_entropy(self):
        df_next = df.loc[df.Outlook != 'overcast']
        df_sunny = df_next.loc[df_next.Outlook == 'sunny']
        entropy_decision = f_entropy_decision(df_sunny)
        self.assertAlmostEqual(entropy_decision, 0.97095059)

    def test_sunny_outlook_info_gain(self):
        df_next = df.loc[df.Outlook != 'overcast']
        df_sunny = df_next.loc[df_next.Outlook == 'sunny']
        entropy_decision = f_entropy_decision(df_sunny)
        self.assertAlmostEqual(IG(df_sunny, 'Temperature', entropy_decision), 0.82809345)
        self.assertAlmostEqual(IG(df_sunny, 'Humidity', entropy_decision), 0.97095059)
        self.assertAlmostEqual(IG(df_sunny, 'Windy', entropy_decision), 0.63131577)


if __name__ == '__main__':
    unittest.main()
