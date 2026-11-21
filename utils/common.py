import pandas as pd


def data_preprocessing(file_path):
    # 1.加载数据
    data = pd.read_csv(file_path)

    # 2.将time列转为datatime类型
    data["time"] = pd.to_datetime(data["time"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    data = data.sort_values(by="time", ascending=True).drop_duplicates()  # 排序与去重

    # 3.返回数据
    return data


def to_formatted_time(start_time, end_time):
    # 计算时间差
    delta = end_time - start_time

    # 提取总秒数
    total_seconds = delta.total_seconds()

    # 计算时、分、秒
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = total_seconds % 60

    # 格式化为 %h%m%s 样式
    formatted_time = f"{hours}h{minutes}m{seconds:.2f}s"
    return formatted_time


if __name__ == "__main__":
    data = data_preprocessing('../data/train.csv')
    print(data)
