import unittest
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def preprocess(data):
    def fix_age(age):
        if np.isnan(age):
            return -1
        elif age < 1:
            return age * 100
        else:
            return age

    data.loc[:, 'Gender'] = data.Gender.apply(lambda s: int(s == 'female'))
    data.loc[:, 'Age'] = data.Age.apply(fix_age)

    embarked = pd.get_dummies(data.Embarked, prefix='Emb')[['Emb_C', 'Emb_Q', 'Emb_S']]
    cols = ['Pclass', 'Gender', 'Age', 'SibSp', 'Parch', 'Fare']

    return pd.concat([data[cols], embarked], axis=1).values


data = pd.read_csv(os.path.join(ROOT_DIR, '..', 'titanic.csv'))

train, val = train_test_split(data, test_size=0.2, random_state=11)

x_train = preprocess(train)
y_train = train['Survived'].values

x_val = preprocess(val)
y_val = val['Survived'].values


class TestingExercise36(unittest.TestCase):
    def test_dataframe_shape(self):
        expected_shape = (712, 9)
        actual_shape = x_train.shape
        self.assertEqual(actual_shape, expected_shape)

    def test_bagging(self):
        dt_params = {
            'criterion': 'entropy',
            'random_state': 11
        }
        dt = DecisionTreeClassifier(**dt_params)
        bc_params = {
            'base_estimator': dt,
            'n_estimators': 50,
            'max_samples': 0.5,
            'random_state': 11,
            'n_jobs': -1
        }
        bc = BaggingClassifier(**bc_params)
        dt.fit(x_train, y_train)
        dt_preds_train = dt.predict(x_train)
        dt_preds_val = dt.predict(x_val)
        bc.fit(x_train, y_train)
        bc_preds_train = bc.predict(x_train)
        bc_preds_val = bc.predict(x_val)
        train_accuracy_dt = accuracy_score(y_true=y_train, y_pred=dt_preds_train)
        val_accuracy_dt = accuracy_score(y_true=y_val, y_pred=dt_preds_val)
        train_accuracy_bc = accuracy_score(y_true=y_train, y_pred=bc_preds_train)
        val_accuracy_bc = accuracy_score(y_true=y_val, y_pred=bc_preds_val)
        self.assertAlmostEqual(train_accuracy_dt, 0.98314607)
        self.assertAlmostEqual(val_accuracy_dt, 0.75977654)
        self.assertAlmostEqual(train_accuracy_bc, 0.92837079)
        self.assertAlmostEqual(val_accuracy_bc, 0.86033520)

    def test_random_forest(self):
        rf_params = {
            'n_estimators': 100,
            'criterion': 'entropy',
            'max_features': 0.5,
            'min_samples_leaf': 10,
            'random_state': 11,
            'n_jobs': -1
        }
        rf = RandomForestClassifier(**rf_params)
        rf.fit(x_train, y_train)
        rf_preds_train = rf.predict(x_train)
        rf_preds_val = rf.predict(x_val)
        train_accuracy = accuracy_score(y_true=y_train, y_pred=rf_preds_train)
        val_accuracy = accuracy_score(y_true=y_val, y_pred=rf_preds_val)
        self.assertAlmostEqual(train_accuracy, 0.82303371)
        self.assertAlmostEqual(val_accuracy, 0.86033520)

    def test_adaboost(self):
        dt_params = {
            'max_depth': 1,
            'random_state': 11
        }
        dt = DecisionTreeClassifier(**dt_params)

        ab_params = {
            'n_estimators': 100,
            'base_estimator': dt,
            'random_state': 11
        }
        ab = AdaBoostClassifier(**ab_params)
        ab.fit(x_train, y_train)
        ab_preds_train = ab.predict(x_train)
        ab_preds_val = ab.predict(x_val)
        train_accuracy = accuracy_score(y_true=y_train, y_pred=ab_preds_train)
        val_accuracy = accuracy_score(y_true=y_val, y_pred=ab_preds_val)
        self.assertAlmostEqual(train_accuracy, 0.82443820)
        self.assertAlmostEqual(val_accuracy, 0.85474860)

    def test_gradboost(self):
        gbc_params = {
            'n_estimators': 100,
            'max_depth': 3,
            'min_samples_leaf': 5,
            'random_state': 11
        }
        gbc = GradientBoostingClassifier(**gbc_params)
        gbc.fit(x_train, y_train)
        gbc_preds_train = gbc.predict(x_train)
        gbc_preds_val = gbc.predict(x_val)
        train_accuracy = accuracy_score(y_true=y_train, y_pred=gbc_preds_train)
        val_accuracy = accuracy_score(y_true=y_val, y_pred=gbc_preds_val)
        self.assertAlmostEqual(train_accuracy, 0.89466292)
        self.assertAlmostEqual(val_accuracy, 0.87709497)

    def test_stacking(self):
        x_train_with_metapreds = np.zeros((x_train.shape[0], x_train.shape[1] + 2))
        x_train_with_metapreds[:, :-2] = x_train
        x_train_with_metapreds[:, -2:] = -1
        self.assertEqual(x_train_with_metapreds[0][0], 3)

        kf = KFold(n_splits=5, random_state=11)
        for train_indices, val_indices in kf.split(x_train):
            kfold_x_train, kfold_x_val = x_train[train_indices], x_train[val_indices]
            kfold_y_train, kfold_y_val = y_train[train_indices], y_train[val_indices]
            svm = LinearSVC(random_state=11, max_iter=1000)
            svm.fit(kfold_x_train, kfold_y_train)
            svm_pred = svm.predict(kfold_x_val)
            knn = KNeighborsClassifier(n_neighbors=4)
            knn.fit(kfold_x_train, kfold_y_train)
            knn_pred = knn.predict(kfold_x_val)
            x_train_with_metapreds[val_indices, -2] = svm_pred
            x_train_with_metapreds[val_indices, -1] = knn_pred

        x_val_with_metapreds = np.zeros((x_val.shape[0], x_val.shape[1] + 2))
        x_val_with_metapreds[:, :-2] = x_val
        x_val_with_metapreds[:, -2:] = -1
        self.assertEqual(x_val_with_metapreds[0][0], 3)

        svm = LinearSVC(random_state=11, max_iter=1000)
        svm.fit(x_train, y_train)
        knn = KNeighborsClassifier(n_neighbors=4)
        knn.fit(x_train, y_train)
        svm_pred = svm.predict(x_val)
        knn_pred = knn.predict(x_val)
        x_val_with_metapreds[:, -2] = svm_pred
        x_val_with_metapreds[:, -1] = knn_pred

        lr = LogisticRegression(random_state=11)
        lr.fit(x_train_with_metapreds, y_train)
        lr_preds_train = lr.predict(x_train_with_metapreds)
        lr_preds_val = lr.predict(x_val_with_metapreds)

        train_accuracy_st = accuracy_score(y_true=y_train, y_pred=lr_preds_train)
        val_accuracy_st = accuracy_score(y_true=y_val, y_pred=lr_preds_val)
        train_accuracy_kn = accuracy_score(y_true=y_train, y_pred=svm.predict(x_train))
        val_accuracy_kn = accuracy_score(y_true=y_val, y_pred=svm_pred)
        train_accuracy_sv = accuracy_score(y_true=y_train, y_pred=knn.predict(x_train))
        val_accuracy_sv = accuracy_score(y_true=y_val, y_pred=knn_pred)
        self.assertAlmostEqual(train_accuracy_st, 0.78651685)
        self.assertAlmostEqual(val_accuracy_st, 0.88826816)
        self.assertAlmostEqual(train_accuracy_kn, 0.69241573)
        self.assertAlmostEqual(val_accuracy_kn, 0.76536313)
        self.assertAlmostEqual(train_accuracy_sv, 0.79213483)
        self.assertAlmostEqual(val_accuracy_sv, 0.67039106)


if __name__ == '__main__':
    unittest.main()
