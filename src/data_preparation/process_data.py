import numpy as np
import pandas as pd
import gc
from sklearn.preprocessing import LabelEncoder
import pandas as pd

from src.data_preparation.create_matrix import create_matrix
from src.data_preparation.load_data import load_data
from src.feature_engineering.add_sales_features import add_sales_features
from src.feature_engineering.campaign_revenue_features import create_campaign_revenue_features
from src.feature_engineering.feature_engineering import add_group_stats, lag_feature, target_encoding
from src.feature_engineering.revenue_features import create_revenue_features
from src.visualization.sales_visualization import plot_sales_by_month


def process_data(data):
    # Xóa các cột không cần thiết
    data.drop(['item_campaign_last_sale', 'item_cnt_month_lag_12', 'date_campaign_avg_item_cnt_lag_12', 'date_item_avg_item_cnt_lag_12'], axis=1, inplace=True)
    
    # Khởi tạo LabelEncoder
    label_encoder = LabelEncoder()

    # Áp dụng LabelEncoder cho từng trường dữ liệu cần được mã hóa
    for column in ['name', 'short_description', 'categories_name']:
        data[column] = label_encoder.fit_transform(data[column])
    
    # Xác định cột chứa giá trị NaN hoặc inf
    invalid_columns = np.unique(np.where(~np.isfinite(data))[1])
    invalid_column_names = data.columns[invalid_columns]
    
    # Chuyển các cột sang dạng số
    data[invalid_column_names] = data[invalid_column_names].astype(float)

    # Chuyển các giá trị inf thành nan
    data.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Thay thế các giá trị nan bằng giá trị trung bình của cột tương ứng
    data.fillna(data.mean(), inplace=True)
    
    # Tách dữ liệu thành tập huấn luyện, kiểm tra và kiểm định
    X_train = data[data.date_block_num < 25].drop(['item_cnt_month'], axis=1)
    Y_train = data[data.date_block_num < 25]['item_cnt_month']
    X_valid = data[data.date_block_num == 25].drop(['item_cnt_month'], axis=1)
    Y_valid = data[data.date_block_num == 25]['item_cnt_month']
    X_test = data[data.date_block_num == 26].drop(['item_cnt_month'], axis=1)
    Y_test = data[data.date_block_num == 26]['item_cnt_month']

    # Thu gom rác để giải phóng bộ nhớ
    gc.collect()
    
    # Chuyển các giá trị inf thành nan trong tập huấn luyện
    X_train.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Thay thế các giá trị nan bằng giá trị trung bình của cột tương ứng trong tập huấn luyện
    X_train.fillna(X_train.mean(), inplace=True)
    
    return X_train, Y_train, X_valid, Y_valid, X_test, Y_test

def prepare_features_and_data():
    # Step 1: Load data
    items, train, test, marketing = load_data()

    # Step 2: Data Visualization
    plot_sales_by_month(train)

    # Bước 1: Tạo ma trận đặc trưng ban đầu
    matrix = create_matrix()

    # Bước 2: Tạo các đặc trưng trễ (lag features)
    matrix = lag_feature(matrix, [1, 2, 3, 6, 12], 'item_cnt_month')

    # Bước 3: Thêm các thống kê nhóm
    matrix = add_group_stats(matrix, ['date_block_num', 'product_id'], 'item_cnt_month', 'product', [6, 12])
    matrix = add_group_stats(matrix, ['date_block_num', 'campaign_id'], 'item_cnt_month', 'campaign', [6, 12])
    matrix = add_group_stats(matrix, ['date_block_num', 'categories_id'], 'item_cnt_month', 'category', [12])

    # Bước 4: Mã hóa mục tiêu (target encoding)
    matrix = target_encoding(matrix, ['date_block_num'], 'item_cnt_month', 'date_avg_item_cnt', [1])
    matrix = target_encoding(matrix, ['date_block_num', 'campaign_id'], 'item_cnt_month', 'date_campaign_avg_item_cnt', [1, 2, 3, 6, 12])
    matrix = target_encoding(matrix, ['date_block_num', 'product_id'], 'item_cnt_month', 'date_item_avg_item_cnt', [1, 2, 3, 6, 12])
    matrix = target_encoding(matrix, ['date_block_num', 'campaign_id', 'categories_id'], 'item_cnt_month', 'date_campaign_cat_avg_item_cnt', [1])

    # Bước 5: Tạo các đặc trưng doanh thu
    matrix = create_revenue_features(matrix, train)
    matrix = create_campaign_revenue_features(matrix, train)

    # Bước 6: Thêm các đặc trưng doanh số
    matrix = add_sales_features(matrix)

    # Bước 7: Xử lý dữ liệu để tạo tập huấn luyện, kiểm tra và kiểm định
    X_train, Y_train, X_valid, Y_valid, X_test, Y_test = process_data(matrix)

    return X_train, Y_train, X_valid, Y_valid, X_test, Y_test





