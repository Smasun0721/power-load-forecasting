import os
import pandas as pd
import datetime

from utils.log import Logger
from utils.common import data_preprocessing, to_formatted_time
from sklearn.metrics import mean_absolute_error
import matplotlib.ticker as mick
import joblib
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "SimHei"
plt.rcParams["font.size"] = 15


class PowerLoadPredict:
    # 初始化属性
    def __init__(self, file_path):
        # 初始化日志对象
        log_name = "predict_" + datetime.datetime.now().strftime("%Y%m%d")
        self.logger = Logger("../", log_name).get_logger()
        self.logger.info("-" * 48)
        self.logger.info("日志系统初始化完成!")
        # 初始化数据
        self.data_source = data_preprocessing(file_path)
        self.logger.info("电力负荷模型预测对象初始化完成！")
        # 将 日期 与 负荷 转化为字典
        self.time_load_dict = self.data_source.set_index("time")["power_load"].to_dict()

    # 解析特征
    def pred_feature_extract(self, time_load_dict, pre_time):
        """
        将现有数据转化成模型训练时所用的数据格式
        :param time_load_dict: 当前需要预测的时间点之前的所有的历史数据
        :param pre_time: 当前需要预测的时间点
        :return:
        """
        feature_names = [
            "hour_0",
            "hour_1",
            "hour_2",
            "hour_3",
            "hour_4",
            "hour_5",
            "hour_6",
            "hour_7",
            "hour_8",
            "hour_9",
            "hour_10",
            "hour_11",
            "hour_12",
            "hour_13",
            "hour_14",
            "hour_15",
            "hour_16",
            "hour_17",
            "hour_18",
            "hour_19",
            "hour_20",
            "hour_21",
            "hour_22",
            "hour_23",
            "month_1",
            "month_2",
            "month_3",
            "month_4",
            "month_5",
            "month_6",
            "month_7",
            "month_8",
            "month_9",
            "month_10",
            "month_11",
            "month_12",
            "前1小时",
            "前2小时",
            "前3小时",
            "yesterday_load",
        ]

        # 提取hour特征
        pre_hour = int(pre_time[11:13])
        hour_list = [1 if pre_hour == i else 0 for i in range(24)]

        # 提取month特征
        pre_month = int(pre_time[5:7])
        month_list = [1 if pre_month == i else 0 for i in range(12)]

        # 提取窗口时间特征
        # 前1小时
        last_1h_load = time_load_dict.get(
            (pd.to_datetime(pre_time) - pd.to_timedelta("1h")).strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            500,
        )

        # 前2小时
        last_2h_load = time_load_dict.get(
            (pd.to_datetime(pre_time) - pd.to_timedelta("2h")).strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            500,
        )

        # 前3小时
        last_3h_load = time_load_dict.get(
            (pd.to_datetime(pre_time) - pd.to_timedelta("3h")).strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            500,
        )

        # 前1天
        yesterday_load = time_load_dict.get(
            (pd.to_datetime(pre_time) - pd.to_timedelta("1D")).strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            500,
        )

        # 拼接特征数据
        feature_data = (
            hour_list
            + month_list
            + [last_1h_load, last_2h_load, last_3h_load, yesterday_load]
        )

        # 转成df对象并返回
        return pd.DataFrame([feature_data], columns=feature_names)

    # 可视化预测结果
    def prediction_plot(self, evaluate_df):
        fig = plt.figure(figsize=(20, 80))
        # 创建子图
        ax = fig.add_subplot()
        # 填充数据
        ax.plot(
            evaluate_df["预测时间"],
            evaluate_df["预测结果"],
            color="blue",
            label="预测结果",
        )
        ax.plot(
            evaluate_df["预测时间"],
            evaluate_df["真实结果"],
            color="red",
            label="真实结果",
        )
        # 添加标题
        ax.set_title("真实负荷与预测负荷关系图")
        ax.set_xlabel("时间")
        ax.set_ylabel("负荷")
        # 调整刻度
        ax.xaxis.set_major_locator(mick.MultipleLocator(base=50))
        # 添加网格
        ax.grid(True)
        # 启用图例
        ax.legend()
        # 保存图片
        plt.savefig("../data/fig/真实负荷与预测负荷关系图.png")
        # 默认只保存图片，设置 SHOW_PLOTS=1 时才弹窗
        if os.environ.get("SHOW_PLOTS") == "1":
            plt.show()

    # 预测
    def predict_test(self):
        # 加载模型
        estimator = joblib.load("../model/xgb_20260919-205459.pkl")

        # 确定要预测的时间段
        pre_times = self.data_source["time"][
            self.data_source["time"] >= "2015-08-01 00:00:00"
        ]

        # 定义evaluate_list列表记录预测结果
        evaluate_list = []
        self.logger.info("开始解析特征并预测...")
        start_time = datetime.datetime.now()
        for pre_time in pre_times:
            # 逐小时获取后期需预测的时间段字典
            time_load_dict_masked = {
                k: v for k, v in self.time_load_dict.items() if k < pre_time
            }

            # 解析特征
            feature_df = self.pred_feature_extract(time_load_dict_masked, pre_time)

            # 记录真实值
            true_value = self.time_load_dict.get(pre_time, 500)

            # 利用加载的模型预测负荷
            y_pred = estimator.predict(feature_df)
            evaluate_list.append(
                [pre_time, true_value, y_pred[0]]
            )  # 预测时间,真实值,预测值
        end_time = datetime.datetime.now()
        self.logger.info(
            f"特征解析与预测完毕! 耗时{to_formatted_time(start_time, end_time)}"
        )

        # 循环结束后,将evaluate_list转换成DateFrame
        evaluate_df = pd.DataFrame(
            evaluate_list, columns=["预测时间", "真实结果", "预测结果"]
        )

        # 计算预测结果与真实值的MAE
        print(
            f"平均绝对误差:{mean_absolute_error(evaluate_df['真实结果'], evaluate_df['预测结果'])}"
        )

        self.prediction_plot(evaluate_df)


# 测试
if __name__ == "__main__":
    pp = PowerLoadPredict("../data/test.csv")
    pp.predict_test()
