import pandas as pd
import numpy as np
import time

def create_revenue_features(matrix, train):
    ts = time.time()

    group = train.groupby(['product_id']).agg({'price': ['mean']})
    group.columns = ['item_avg_item_price']
    group.reset_index(inplace=True)

    matrix = pd.merge(matrix, group, on=['product_id'], how='left')
    matrix['item_avg_item_price'] = matrix['item_avg_item_price'].astype(np.float16)

    group = train.groupby(['date_block_num','product_id']).agg({'price': ['mean']})
    group.columns = ['date_item_avg_item_price']
    group.reset_index(inplace=True)

    matrix = pd.merge(matrix, group, on=['date_block_num','product_id'], how='left')
    matrix['date_item_avg_item_price'] = matrix['date_item_avg_item_price'].astype(np.float16)

    lags = [1,2,3,4,5,6]
    for lag in lags:
        matrix['date_item_avg_item_price_lag_'+str(lag)] = matrix.groupby(['product_id'])['date_item_avg_item_price'].shift(lag)

    for lag in lags:
        matrix['delta_price_lag_'+str(lag)] = \
            (matrix['date_item_avg_item_price_lag_'+str(lag)] - matrix['item_avg_item_price']) / matrix['item_avg_item_price']

    def select_trend(row):
        for lag in lags:
            if pd.notna(row['delta_price_lag_'+str(lag)]):
                return row['delta_price_lag_'+str(lag)]
        return 0

    matrix['delta_price_lag'] = matrix.apply(select_trend, axis=1)
    matrix['delta_price_lag'] = matrix['delta_price_lag'].astype(np.float16)
    matrix['delta_price_lag'].fillna(0, inplace=True)

    features_to_drop = ['item_avg_item_price', 'date_item_avg_item_price']
    for lag in lags:
        features_to_drop += ['date_item_avg_item_price_lag_'+str(lag)]
        features_to_drop += ['delta_price_lag_'+str(lag)]

    matrix.drop(features_to_drop, axis=1, inplace=True)

    time.time() - ts

    return matrix
