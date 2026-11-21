import os
import pandas as pd
import matplotlib.pyplot as plt
import datetime

from utils.log import Logger
from utils.common import data_preprocessing, to_formatted_time
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    root_mean_squared_error,
    mean_absolute_percentage_error,
)
import joblib

# 解决matplotlib中文显示异常问题
plt.rcParams["font.family"] = "SimHei"
plt.rcParams["font.size"] = 15


# 创建电力负荷预测模型类
class PowerLoadModel:
    # 初始化属性
    def __init__(self,file_path):
        # 初始化日志对象
        log_name = "train_" + datetime.datetime.now().strftime("%Y%m%d")
        self.logger = Logger("../", log_name).get_logger()
        self.logger.info("-" * 48)
        self.logger.info("日志系统初始化完成!")
        # 初始化数据
        self.data_source = data_preprocessing(file_path)
        self.logger.info("电力负荷模型训练对象初始化完成！")

    # 可视化数据分析
    def ana_data(self):
        self.logger.info("开始进行数据可视化分析...")
        # 复制数据
        ana_data = self.data_source.copy()
        # 创建画布
        fig = plt.figure(figsize=(30, 40))

        # 添加子图一:负荷整体分布情况
        self.logger.info("绘制子图一...")
        ax1 = fig.add_subplot(411)
        ax1.hist(ana_data["power_load"], bins=100)  # 电力负荷直方图
        ax1.set_title("负荷整体分布情况")
        ax1.set_xlabel("负荷")
        ax1.set_ylabel("频率")

        # 添加子图二:一天中各小时平均负荷变化趋势
        ana_data["hour"] = pd.to_datetime(ana_data["time"]).dt.hour  # 添加hour列
        hour_load_mean = ana_data.groupby(ana_data["hour"], as_index=False)[
            "power_load"
        ].mean()  # 统计各小时的平均负荷
        self.logger.info("绘制子图二...")
        ax2 = fig.add_subplot(412)
        ax2.plot(hour_load_mean["hour"], hour_load_mean["power_load"])  # 折线图
        ax2.set_title("各小时的平均负荷趋势")
        ax2.set_xlabel("小时/h")
        ax2.set_ylabel("负荷")
        ax2.set_xticks(hour_load_mean["hour"])

        # 添加子图三:各个月份的平均负荷
        ana_data["month"] = pd.to_datetime(ana_data["time"]).dt.month  # 添加month列
        month_load_mean = ana_data.groupby(ana_data["month"], as_index=False)[
            "power_load"
        ].mean()  # 统计各月份的平均负荷
        self.logger.info("绘制子图三...")
        ax3 = fig.add_subplot(413)
        ax3.plot(month_load_mean["month"], month_load_mean["power_load"])  # 折线图
        ax3.set_title("各month的平均负荷趋势")
        ax3.set_xlabel("月份/m")
        ax3.set_ylabel("负荷")
        ax3.set_xticks(month_load_mean["month"])

        # 添加子图四:工作日与周末的平均负荷情况
        ana_data["weekday"] = ana_data["time"].apply(
            lambda x: pd.to_datetime(x).weekday()
        )
        ana_data["is_holiday"] = ana_data["weekday"].apply(
            lambda x: True if x in [5, 6] else False
        )
        holiday_load_mean = ana_data[ana_data["is_holiday"] == True][
            "power_load"
        ].mean()
        weekday_load_mean = ana_data[ana_data["is_holiday"] == False][
            "power_load"
        ].mean()
        self.logger.info("绘制子图四...")
        ax4 = fig.add_subplot(414)
        ax4.bar(["工作日", "周末"], [weekday_load_mean, holiday_load_mean])
        ax4.set_title("工作日与周末的平均负荷")
        ax4.set_ylabel("负荷")

        # self.logger.info("打印分析后数据...")
        # print(ana_data)
        plt.savefig("../data/fig/负荷整体的分布情况.png")
        self.logger.info("保存图片成功!")
        if os.environ.get("SHOW_PLOTS") == "1":
            plt.show()
        self.logger.info("数据可视化分析结束!")

    # 特征工程
    def feature_engineering(self):
        self.logger.info("开始进行特征工程...")
        # 复制数据
        feature_data = self.data_source.copy()

        # 1.提取出时间特征:小时、月份
        self.logger.info("提取出时间特征:小时、月份")
        feature_data["hour"] = pd.to_datetime(feature_data["time"]).dt.hour
        feature_data["month"] = pd.to_datetime(feature_data["time"]).dt.month
        # 热编码 one-hot 处理 month、hour 字段
        feature_data = pd.get_dummies(feature_data, columns=["hour", "month"])

        # 2.提取出相近时间窗口中的负荷特征，step大小窗口的负荷
        self.logger.info("提取出相近时间窗口中的负荷特征:前1小时,前2小时,前3小时")
        before_1h_load = feature_data["power_load"].shift(1)
        before_2h_load = feature_data["power_load"].shift(2)
        before_3h_load = feature_data["power_load"].shift(3)
        before_load = pd.concat(
            [before_1h_load, before_2h_load, before_3h_load], axis=1
        )
        before_load.columns = ["前1小时", "前2小时", "前3小时"]
        feature_data = pd.concat([feature_data, before_load], axis=1)

        # 3.提取昨日同时刻负荷特征
        self.logger.info("提取昨日同时刻负荷特征")
        yesterday_dt = pd.to_datetime(feature_data["time"]) - pd.Timedelta(days=1)
        # 将时间转换为字符串格式
        feature_data["yesterday_time"] = yesterday_dt.dt.strftime("%Y-%m-%d %H:%M:%S")
        # 将所有 日期 与 负荷 组成字典，方便查找
        time_load_dict = feature_data.set_index("time")["power_load"].to_dict()
        feature_data["yesterday_load"] = feature_data["yesterday_time"].map(
            time_load_dict
        )

        # 4.剔除含有空值的样本
        self.logger.info("剔除含有空值的样本")
        feature_data = feature_data.dropna()

        # 5.整理特征列
        self.logger.info("整理特征列")
        feature_columns = (
            list(feature_data.iloc[:, 1:38].columns)  # 小时、月份
            + list(before_load.columns)  # "前1小时", "前2小时", "前3小时"
            + ["yesterday_load"]  # 昨日同时刻负荷特征
        )

        # 6.返回数据
        self.logger.info("特征工程结束!")
        return feature_data[feature_columns]

    # 模型训练与评估
    def model_train(self, data):
        self.logger.info("开始进行模型训练...")
        start_time = datetime.datetime.now()
        x = data.iloc[:, 1:]
        y = data.iloc[:, 0]
        print(x)
        print(y)

        # 数据集切分
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, random_state=0
        )

        # # 网格搜索 + 交叉验证
        # self.logger.info("开始进行网格搜索 + 交叉验证")
        # start_time = datetime.datetime.now()
        # self.logger.info(f"开始时间{start_time}")
        # # 创建参数字典
        # params_dict = {
        #     "n_estimators": [400, 500, 600],
        #     "learning_rate": [0.025, 0.03, 0.035, 0.04],
        #     "max_depth": [5, 6, 7],
        # }
        #
        # # 创建XGBoost对象
        # estimator = XGBRegressor()
        #
        # # 创建网格搜索对象
        # gs = GridSearchCV(estimator, param_grid=params_dict, n_jobs=-4, cv=5)
        # gs.fit(x_train, y_train)
        #
        # end_time = datetime.datetime.now()
        # self.logger.info(f"结束时间:{end_time}")
        # self.logger.info(f"网格搜索耗时 {to_formatted_time(start_time, end_time)} s")
        #
        # # 打印最优参数组合
        # print(f"最优参数组合:{gs.best_params_}")

        # 找到最优超参组合后
        estimator = XGBRegressor(
            n_estimators=500, max_depth=6, learning_rate=0.03, n_jobs=-4
        )
        estimator.fit(x_train, y_train)

        self.logger.info("模型训练结束!")
        end_time = datetime.datetime.now()
        self.logger.info(f"模型训练总耗时:{to_formatted_time(start_time, end_time)}")

        self.logger.info("模型评估开始...")
        # 模型预测
        y_pred = estimator.predict(x_test)
        # 模型评估
        print(estimator.score(x_test, y_test))
        print(f"均方误差:{mean_squared_error(y_test, y_pred)}")
        print(f"均方根误差:{root_mean_squared_error(y_test, y_pred)}")
        print(f"平均绝对误差:{mean_absolute_error(y_test, y_pred)}")
        print(f"平均绝对百分比误差:{mean_absolute_percentage_error(y_test, y_pred)}")

        self.logger.info("模型评估结束!")

        # 模型保存
        now_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # 获取当前时间
        joblib.dump(estimator, f"../model/xgb_{now_str}.pkl")
        self.logger.info(
            f"模型已保存! 保存路径:{os.path.abspath(f'../model/xgb_{now_str}.pkl')}"
        )


# 测试
if __name__ == "__main__":
    # 实例化对象
    pm = PowerLoadModel('../data/train.csv')
    # 可视化数据分析
    # pm.ana_data()

    # 特征工程
    feature_data = pm.feature_engineering()
    print(feature_data)

    # 模型训练
    pm.model_train(feature_data)
