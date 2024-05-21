import pandas as pd
import numpy as np
from src.feature_engineering.feature_engineering import lag_feature

def create_campaign_revenue_features(matrix, train):

    train['revenue'] = train['price'] -train['purchase_price']
    train['profit'] = train['revenue']*train['orders']
    # Tính tổng doanh thu của mỗi cửa hàng theo tháng và gán vào cột mới 'date_shop_revenue'
    group = train.groupby(['date_block_num','campaign_id']).agg({'revenue': ['sum']})
    group.columns = ['date_campaign_revenue']
    group.reset_index(inplace=True)

    # Kết hợp DataFrame 'matrix' với DataFrame 'group' dựa trên các cột 'date_block_num' và 'shop_id'
    matrix = pd.merge(matrix, group, on=['date_block_num','campaign_id'], how='left')
    matrix['date_campaign_revenue'] = matrix['date_campaign_revenue'].astype(np.float32)

    # Tính giá trị trung bình của doanh thu của mỗi cửa hàng và gán vào cột mới 'shop_avg_revenue'
    group = group.groupby(['campaign_id']).agg({'date_campaign_revenue': ['mean']})
    group.columns = ['campaign_avg_revenue']
    group.reset_index(inplace=True)

    # Kết hợp DataFrame 'matrix' với DataFrame 'group' dựa trên cột 'shop_id'
    matrix = pd.merge(matrix, group, on=['campaign_id'], how='left')
    matrix['campaign_avg_revenue'] = matrix['campaign_avg_revenue'].astype(np.float32)

    # Tính toán biến thể của doanh thu so với giá trị trung bình của mỗi cửa hàng và gán vào cột mới 'delta_revenue'
    matrix['delta_revenue'] = (matrix['date_campaign_revenue'] - matrix['campaign_avg_revenue']) / matrix['campaign_avg_revenue']
    matrix['delta_revenue'] = matrix['delta_revenue'].astype(np.float16)

    # Tạo đặc trưng độ trễ từ cột 'delta_revenue' sử dụng độ trễ là 1
    matrix = lag_feature(matrix, [1], 'delta_revenue')

    # Loại bỏ các cột không cần thiết đã tạo ra
    matrix.drop(['date_campaign_revenue','campaign_avg_revenue','delta_revenue'], axis=1, inplace=True)

    return matrix
