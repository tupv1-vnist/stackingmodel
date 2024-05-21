import matplotlib.pyplot as plt

def plot_sales_by_month(train):
    sale_by_month = train.groupby('date_block_num')['orders'].sum()

    plt.figure(figsize=(10, 6))
    plt.plot(sale_by_month.index, sale_by_month.values, marker='o', linestyle='-')
    plt.title('Tổng doanh số theo tháng')
    plt.xlabel('Tháng')
    plt.ylabel('Doanh số')
    plt.grid(True)
    plt.show()
