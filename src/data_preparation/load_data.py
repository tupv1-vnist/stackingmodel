import os
import pandas as pd

def load_data():
    # Xác định đường dẫn tuyệt đối tới thư mục chứa tệp dữ liệu
    base_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_path, '../../data')

    # Đọc các tệp CSV
    items = pd.read_csv(os.path.join(data_path, 'item1.csv'))
    train = pd.read_csv(os.path.join(data_path, 'order1.csv'))
    test = pd.read_csv(os.path.join(data_path, 'test.csv'))
    marketing = pd.read_csv(os.path.join(data_path, 'marketing.csv'))

    # Convert 'date' column to datetime
    train['date'] = pd.to_datetime(train['date'])

    # Create 'date_block_num' column based on 'date'
    train['date_block_num'] = (train['date'].dt.year - 2021) * 12 + train['date'].dt.month

    return items, train, test,marketing
