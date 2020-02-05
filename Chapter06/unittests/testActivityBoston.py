import unittest
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

data = pd.read_csv(os.path.join(ROOT_DIR, '..', 'Datasets', 'boston_house_prices.csv'))


class TestingActivityBoston(unittest.TestCase):
    def test_dataset_shape(self):
        self.assertEqual(data.shape, (506, 14))

    def test_classifier_scores(self):
        data_final = data.fillna(-1)
        train, val = train_test_split(data_final, test_size=0.2, random_state=11)

        x_train = train.drop(columns=['PRICE'])
        y_train = train['PRICE'].values

        x_val = val.drop(columns=['PRICE'])
        y_val = val['PRICE'].values

        train_mae_values, val_mae_values = {}, {}

        # Decision Tree

        dt_params = {
            'criterion': 'mae',
            'min_samples_leaf': 15,
            'random_state': 11
        }

        dt = DecisionTreeRegressor(**dt_params)

        dt.fit(x_train, y_train)
        dt_preds_train = dt.predict(x_train)
        dt_preds_val = dt.predict(x_val)

        train_mae_values['dt'] = mean_absolute_error(y_true=y_train, y_pred=dt_preds_train)
        val_mae_values['dt'] = mean_absolute_error(y_true=y_val, y_pred=dt_preds_val)

        # k-Nearest Neighbours

        knn_params = {
            'n_neighbors': 5
        }

        knn = KNeighborsRegressor(**knn_params)

        knn.fit(x_train, y_train)
        knn_preds_train = knn.predict(x_train)
        knn_preds_val = knn.predict(x_val)

        train_mae_values['knn'] = mean_absolute_error(y_true=y_train, y_pred=knn_preds_train)
        val_mae_values['knn'] = mean_absolute_error(y_true=y_val, y_pred=knn_preds_val)

        # Random Forest

        rf_params = {
            'n_estimators': 20,
            'criterion': 'mae',
            'max_features': 'sqrt',
            'min_samples_leaf': 10,
            'random_state': 11,
            'n_jobs': -1
        }

        rf = RandomForestRegressor(**rf_params)

        rf.fit(x_train, y_train)
        rf_preds_train = rf.predict(x_train)
        rf_preds_val = rf.predict(x_val)

        train_mae_values['rf'] = mean_absolute_error(y_true=y_train, y_pred=rf_preds_train)
        val_mae_values['rf'] = mean_absolute_error(y_true=y_val, y_pred=rf_preds_val)

        # Gradient Boosting

        gbr_params = {
            'n_estimators': 20,
            'criterion': 'mae',
            'max_features': 'sqrt',
            'max_depth': 3,
            'min_samples_leaf': 10,
            'random_state': 11
        }

        gbr = GradientBoostingRegressor(**gbr_params)

        gbr.fit(x_train, y_train)
        gbr_preds_train = gbr.predict(x_train)
        gbr_preds_val = gbr.predict(x_val)

        train_mae_values['gbr'] = mean_absolute_error(y_true=y_train, y_pred=gbr_preds_train)
        val_mae_values['gbr'] = mean_absolute_error(y_true=y_val, y_pred=gbr_preds_val)

        # stacking model

        num_base_predictors = len(train_mae_values)  # 4

        x_train_with_metapreds = np.zeros((x_train.shape[0], x_train.shape[1] + num_base_predictors))
        x_train_with_metapreds[:, :-num_base_predictors] = x_train
        x_train_with_metapreds[:, -num_base_predictors:] = -1

        kf = KFold(n_splits=5, random_state=11)

        for train_indices, val_indices in kf.split(x_train):
            kfold_x_train, kfold_x_val = x_train.iloc[train_indices], x_train.iloc[val_indices]
            kfold_y_train, kfold_y_val = y_train[train_indices], y_train[val_indices]

            predictions = []

            dt = DecisionTreeRegressor(**dt_params)
            dt.fit(kfold_x_train, kfold_y_train)
            predictions.append(dt.predict(kfold_x_val))

            knn = KNeighborsRegressor(**knn_params)
            knn.fit(kfold_x_train, kfold_y_train)
            predictions.append(knn.predict(kfold_x_val))

            gbr = GradientBoostingRegressor(**gbr_params)
            gbr.fit(kfold_x_train, kfold_y_train)
            predictions.append(gbr.predict(kfold_x_val))

            for i, preds in enumerate(predictions):
                x_train_with_metapreds[val_indices, -(i + 1)] = preds

        x_val_with_metapreds = np.zeros((x_val.shape[0], x_val.shape[1] + num_base_predictors))
        x_val_with_metapreds[:, :-num_base_predictors] = x_val
        x_val_with_metapreds[:, -num_base_predictors:] = -1

        predictions = []

        dt = DecisionTreeRegressor(**dt_params)
        dt.fit(x_train, y_train)
        predictions.append(dt.predict(x_val))

        knn = KNeighborsRegressor(**knn_params)
        knn.fit(x_train, y_train)
        predictions.append(knn.predict(x_val))

        gbr = GradientBoostingRegressor(**gbr_params)
        gbr.fit(x_train, y_train)
        predictions.append(gbr.predict(x_val))

        for i, preds in enumerate(predictions):
            x_val_with_metapreds[:, -(i + 1)] = preds

        lr = LinearRegression(normalize=True)
        lr.fit(x_train_with_metapreds, y_train)
        lr_preds_train = lr.predict(x_train_with_metapreds)
        lr_preds_val = lr.predict(x_val_with_metapreds)

        train_mae_values['lr'] = mean_absolute_error(y_true=y_train, y_pred=lr_preds_train)
        val_mae_values['lr'] = mean_absolute_error(y_true=y_val, y_pred=lr_preds_val)

        mae_scores = pd.concat([pd.Series(train_mae_values, name='train'),
                                pd.Series(val_mae_values, name='val')],
                               axis=1)

        self.assertAlmostEqual(mae_scores[mae_scores.index == 'dt']['train'].values[0], 2.38440594, places=4)
        self.assertAlmostEqual(mae_scores[mae_scores.index == 'dt']['val'].values[0], 3.28235294, places=4)
        self.assertAlmostEqual(mae_scores[mae_scores.index == 'knn']['train'].values[0], 3.45554455, places=4)
        self.assertAlmostEqual(mae_scores[mae_scores.index == 'knn']['val'].values[0], 3.97803922, places=4)
        self.assertAlmostEqual(mae_scores[mae_scores.index == 'rf']['train'].values[0], 2.31612005, places=4)
        self.assertAlmostEqual(mae_scores[mae_scores.index == 'rf']['val'].values[0], 3.029828431, places=4)
        self.assertAlmostEqual(mae_scores[mae_scores.index == 'gbr']['train'].values[0], 2.46343592, places=4)
        self.assertAlmostEqual(mae_scores[mae_scores.index == 'gbr']['val'].values[0], 3.058634, places=4)
        self.assertAlmostEqual(mae_scores[mae_scores.index == 'lr']['train'].values[0], 2.24627884, places=4)
        self.assertAlmostEqual(mae_scores[mae_scores.index == 'lr']['val'].values[0], 2.87408434, places=4)


if __name__ == '__main__':
    unittest.main()
