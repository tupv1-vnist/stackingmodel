import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from lightgbm import early_stopping

def model_final(X_train, Y_train, X_valid, Y_valid, X_test):
    # Best params cho LightGBM
    best_lgb = {
        'colsample_bytree': 0.8089140355585571,
        'max_depth': 8,
        'min_child_weight': 0,
        'n_estimators': 9,
        'reg_alpha': 0.7633695810217804,
        'reg_lambda': 0.6705049189470987
    }

    # Best params cho XGBoost
    best_xgb = {
        'colsample_bytree': 0.8911958012150187,
        'gamma': 0.0070999104149873475,
        'learning_rate': 0.1961063283600617,
        'max_depth': 8,
        'min_child_weight': 0,
        'n_estimators': 8,
        'subsample': 0.9946360807263804
    }

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
            colsample_bytree=best_lgb['colsample_bytree'],
            max_depth=best_lgb['max_depth'],
            min_child_weight=best_lgb['min_child_weight'],
            n_estimators=best_lgb['n_estimators'],
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
            colsample_bytree=best_xgb['colsample_bytree'],
            gamma=best_xgb['gamma'],
            learning_rate=best_xgb['learning_rate'],
            max_depth=best_xgb['max_depth'],
            min_child_weight=best_xgb['min_child_weight'],
            n_estimators=best_xgb['n_estimators'],
            subsample=best_xgb['subsample'],
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
    # Đường dẫn đến tệp CSV
    output_dir = '../../output/data'
    output_file = os.path.join(output_dir, 'stacked_submission.csv')

    # Tạo thư mục nếu không tồn tại
    os.makedirs(output_dir, exist_ok=True)
    submission.to_csv(output_file, index=False)
    
    return rmse

