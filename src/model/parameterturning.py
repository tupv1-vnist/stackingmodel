import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from lightgbm import early_stopping

def optimize_and_stack(X_train, Y_train, X_valid, Y_valid, X_test):
    # Định nghĩa không gian tham số cho LightGBM và XGBoost
    space_lgb = {
        'max_depth': hp.choice('max_depth', range(3, 12)),
        'n_estimators': hp.choice('n_estimators', range(100, 1001, 100)),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1.0),
        'min_child_weight': hp.choice('min_child_weight', range(100, 601, 100)),
        'reg_alpha': hp.uniform('reg_alpha', 0, 1),
        'reg_lambda': hp.uniform('reg_lambda', 0, 2)
    }

    space_xgb = {
        'max_depth': hp.choice('max_depth', range(3, 12)),
        'n_estimators': hp.choice('n_estimators', range(100, 1001, 100)),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1.0),
        'min_child_weight': hp.choice('min_child_weight', range(100, 601, 100)),
        'subsample': hp.uniform('subsample', 0.5, 1.0),
        'gamma': hp.uniform('gamma', 0, 0.1),
        'learning_rate': hp.uniform('learning_rate', 0.01, 0.2)
    }

    # Hàm mục tiêu để tối ưu hóa LightGBM
    def objective_lgb(space):
        model = LGBMRegressor(
            max_depth=space['max_depth'],
            n_estimators=space['n_estimators'],
            colsample_bytree=space['colsample_bytree'],
            min_child_weight=space['min_child_weight'],
            reg_alpha=space['reg_alpha'],
            reg_lambda=space['reg_lambda'],
            random_state=42,
            verbosity=-1 
        )
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        rmse_scores = []
        for train_index, valid_index in kf.split(X_train):
            X_tr, X_val = X_train.iloc[train_index], X_train.iloc[valid_index]
            y_tr, y_val = Y_train.iloc[train_index], Y_train.iloc[valid_index]
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric='rmse', callbacks=[early_stopping(stopping_rounds=100)])
            preds = model.predict(X_val)
            rmse_scores.append(np.sqrt(mean_squared_error(y_val, preds)))
        return {'loss': np.mean(rmse_scores), 'status': STATUS_OK}

    # Hàm mục tiêu để tối ưu hóa XGBoost
    def objective_xgb(space):
        model = XGBRegressor(
            max_depth=space['max_depth'],
            n_estimators=space['n_estimators'],
            colsample_bytree=space['colsample_bytree'],
            min_child_weight=space['min_child_weight'],
            subsample=space['subsample'],
            gamma=space['gamma'],
            learning_rate=space['learning_rate'],
            random_state=42
        )
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        rmse_scores = []
        for train_index, valid_index in kf.split(X_train):
            X_tr, X_val = X_train.iloc[train_index], X_train.iloc[valid_index]
            y_tr, y_val = Y_train.iloc[train_index], Y_train.iloc[valid_index]
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric='rmse', early_stopping_rounds=50, verbose=False)
            preds = model.predict(X_val)
            rmse_scores.append(np.sqrt(mean_squared_error(y_val, preds)))
        return {'loss': np.mean(rmse_scores), 'status': STATUS_OK}

    # Tối ưu hóa tham số
    trials_lgb = Trials()
    best_lgb = fmin(fn=objective_lgb, space=space_lgb, algo=tpe.suggest, max_evals=100, trials=trials_lgb)

    trials_xgb = Trials()
    best_xgb = fmin(fn=objective_xgb, space=space_xgb, algo=tpe.suggest, max_evals=100, trials=trials_xgb)

    print("Best params for LightGBM:", best_lgb)
    print("Best params for XGBoost:", best_xgb)

    # Sử dụng các tham số tối ưu để huấn luyện và đánh giá mô hình cuối cùng
    model_lgb = LGBMRegressor(**best_lgb)
    model_xgb = XGBRegressor(**best_xgb)

    # Khởi tạo K-Fold cross-validation
    n_folds = 5
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    # Chuẩn bị mảng để lưu trữ dự đoán cho tập valid và test
    valid_preds_lgb = np.zeros(X_valid.shape[0])
    valid_preds_xgb = np.zeros(X_valid.shape[0])
    test_preds_lgb = np.zeros(X_test.shape[0])
    test_preds_xgb = np.zeros(X_test.shape[0])

    # Sử dụng tham số tối ưu đã tìm được để huấn luyện và dự đoán
    for train_index, valid_index in kf.split(X_train):
        X_tr, X_val = X_train.iloc[train_index], X_train.iloc[valid_index]
        y_tr, y_val = Y_train.iloc[train_index], Y_train.iloc[valid_index]

        # LightGBM với tham số tối ưu
        model_lgb = LGBMRegressor(
            max_depth=int(best_lgb['max_depth']),
            n_estimators=int(best_lgb['n_estimators']),
            colsample_bytree=best_lgb['colsample_bytree'],
            min_child_weight=int(best_lgb['min_child_weight']),
            reg_alpha=best_lgb['reg_alpha'],
            reg_lambda=best_lgb['reg_lambda'],
            random_state=42,
            verbosity=-1 
        )
        model_lgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric='rmse', callbacks=[early_stopping(stopping_rounds=100)])
        valid_preds_lgb += model_lgb.predict(X_valid) / n_folds
        test_preds_lgb += model_lgb.predict(X_test) / n_folds

        # XGBoost với tham số tối ưu
        model_xgb = XGBRegressor(
            max_depth=int(best_xgb['max_depth']),
            n_estimators=int(best_xgb['n_estimators']),
            colsample_bytree=best_xgb['colsample_bytree'],
            min_child_weight=int(best_xgb['min_child_weight']),
            subsample=best_xgb['subsample'],
            gamma=best_xgb['gamma'],
            learning_rate=best_xgb['learning_rate'],
            random_state=42
        )
        model_xgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric='rmse', early_stopping_rounds=40)
        valid_preds_xgb += model_xgb.predict(X_valid) / n_folds
        test_preds_xgb += model_xgb.predict(X_test) / n_folds

    # Tạo DataFrame cho các dự đoán từ các mô hình base trên tập valid và test
    valid_level2 = pd.DataFrame({'lgb': valid_preds_lgb, 'xgb': valid_preds_xgb})
    test_level2 = pd.DataFrame({'lgb': test_preds_lgb, 'xgb': test_preds_xgb})

    # Huấn luyện mô hình meta với XGBoost trên tập valid
    meta_model = XGBRegressor(
        max_depth=5,
        n_estimators=100,
        learning_rate=0.05,
        subsample=0.7,
        colsample_bytree=0.7,
        random_state=42
    )
    meta_model.fit(valid_level2, Y_valid)

    # Dự đoán trên tập valid và test sử dụng mô hình meta
    final_preds_valid = meta_model.predict(valid_level2)
    final_preds_test = meta_model.predict(test_level2)

    # Đánh giá mô hình trên tập valid
    rmse = np.sqrt(mean_squared_error(Y_valid, final_preds_valid))
    print('Stacking Model RMSE on Validation data:', rmse)

    # Lưu kết quả cuối cùng vào CSV
    submission = pd.DataFrame({
        'ID': np.arange(final_preds_test.shape[0]),
        'item_cnt_month': final_preds_test
    })
    submission.to_csv('../../output/data/stacked_submission.csv', index=False)
    
    return rmse


