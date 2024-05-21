import pandas as pd
import numpy as np
import time

def add_sales_features(matrix):
    def add_item_campaign_last_sale(matrix):
        ts = time.time()
        last_sale = pd.DataFrame()

        for month in range(1, 37):
            last_month = matrix.loc[(matrix['date_block_num'] < month) & (matrix['item_cnt_month'] > 0)].groupby(['product_id','campaign_id'])['date_block_num'].max()
            
            df = pd.DataFrame({
                'date_block_num': np.ones([last_month.shape[0],])*month,
                'product_id': last_month.index.get_level_values(0).values,
                'campaign_id': last_month.index.get_level_values(1).values,
                'item_campaign_last_sale': last_month.values
            })
            
            last_sale = pd.concat([last_sale, df], ignore_index=True)

        last_sale['date_block_num'] = last_sale['date_block_num'].astype(np.int32)
        matrix = matrix.merge(last_sale, on=['date_block_num','product_id','campaign_id'], how='left')
        print("Time taken for adding item_campaign_last_sale:", time.time() - ts, "seconds.")
        return matrix

    def add_item_last_sale(matrix):
        ts = time.time()
        last_sale = pd.DataFrame()

        for month in range(1, 37):
            last_month = matrix.loc[(matrix['date_block_num'] < month) & (matrix['item_cnt_month'] > 0)].groupby('product_id')['date_block_num'].max()
            
            df = pd.DataFrame({
                'date_block_num': np.ones([last_month.shape[0],])*month,
                'product_id': last_month.index.values,
                'item_last_sale': last_month.values
            })
            
            last_sale = pd.concat([last_sale, df], ignore_index=True)

        last_sale['date_block_num'] = last_sale['date_block_num'].astype(np.int32)
        matrix = matrix.merge(last_sale, on=['date_block_num', 'product_id'], how='left')
        print("Time taken for adding item_last_sale:", time.time() - ts, "seconds.")
        return matrix

    def add_item_campaign_first_sale(matrix):
        ts = time.time()
        matrix['item_campaign_first_sale'] = matrix['date_block_num'] - matrix.groupby(['product_id','campaign_id'])['date_block_num'].transform('min')
        print("Time taken for adding item_campaign_first_sale:", time.time() - ts, "seconds.")
        return matrix

    def add_item_first_sale(matrix):
        ts = time.time()
        matrix['item_first_sale'] = matrix['date_block_num'] - matrix.groupby('product_id')['date_block_num'].transform('min')
        print("Time taken for adding item_first_sale:", time.time() - ts, "seconds.")
        return matrix

    # Thêm thông tin về lần bán hàng cuối cùng cho mỗi cặp cửa hàng / sản phẩm
    matrix = add_item_campaign_last_sale(matrix)

    # Thêm thông tin về lần bán hàng cuối cùng của mỗi sản phẩm
    matrix = add_item_last_sale(matrix)

    # Tính số tháng kể từ lần bán hàng đầu tiên cho mỗi cặp cửa hàng / sản phẩm
    matrix = add_item_campaign_first_sale(matrix)

    # Tính số tháng kể từ lần bán hàng đầu tiên của mỗi sản phẩm
    matrix = add_item_first_sale(matrix)

    return matrix


