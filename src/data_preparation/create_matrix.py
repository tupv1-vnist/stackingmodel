import src.data_preparation.load_data as load_data
import pandas as pd


def create_matrix():
    items, train, test,marketing=load_data.load_data()
    train['revenue'] = train['price'] -train['purchase_price']
    train['profit'] = train['revenue']*train['orders']


    group = train.groupby(['date_block_num','campaign_id','product_id']).agg({'orders': ['sum'], 'profit': ['sum']})

    # Đặt lại tên cột kết quả
    group.columns = ['item_cnt_month', 'profit_month']

    group

    # # Kết hợp DataFrame 'matrix' với DataFrame 'group' dựa trên các cột trong 'cols' và dùng phương pháp 'left join'
    # matrix = matrix.merge(group, on=['date_block_num', 'campaign_id', 'product_id'], how='left')

   

    # Đặt lại index của group
    group = group.reset_index()
    matrix=[]
    # Tạo matrix từ group
    matrix = group.copy()

    # Chuyển đổi các cột về dạng số nguyên
    matrix['date_block_num'] = matrix['date_block_num'].astype(int)
    matrix['campaign_id'] = matrix['campaign_id'].astype(int)
    matrix['product_id'] = matrix['product_id'].astype(int)

    # Sắp xếp matrix theo các cột 'date_block_num', 'campaign_id', và 'product_id'
    matrix.sort_values(['date_block_num', 'campaign_id', 'product_id'], inplace=True)
    # Thực hiện phép nối giữa DataFrame 'matrix' và DataFrame 'items' dựa trên cột 'product_id' của 'matrix' và cột 'id' của 'items'
    matrix = pd.merge(matrix, items, left_on=['product_id'], right_on=['id'], how='left')

    # Xóa cột 'id' trong DataFrame 'matrix' nếu bạn không cần nó sau khi đã thực hiện phép nối
    matrix.drop('id', axis=1, inplace=True)
    return matrix


