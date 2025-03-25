import re

import pandas as pd
from sklearn.model_selection import train_test_split

from action_opt import configs


def text_pretreatment(text):
    # text = re.sub("#.*#", "", text)  # 去除话题
    text = text.lower().strip()  # 去除首尾空格
    text = text[:256]  # 只取前256个字符
    return text


def save_data(X_train, X_test, y_train, y_test):
    train_data = pd.DataFrame({"X": X_train, "y": y_train})
    test_data = pd.DataFrame({"X": X_test, "y": y_test})

    train_data.to_excel(configs.TRAIN_DATA_PATH, index=False)
    test_data.to_excel(configs.VAL_DATA_PATH, index=False)


def load_saved_data():
    train_data = pd.read_excel(configs.TRAIN_DATA_PATH)
    test_data = pd.read_excel(configs.VAL_DATA_PATH)

    X_train = train_data["X"].tolist()
    y_train = train_data["y"].map(eval).tolist()
    X_test = test_data["X"].tolist()
    y_test = test_data["y"].map(eval).tolist()

    return X_train, X_test, y_train, y_test

def load_data():
    """

    :return:
    """

    # data = pd.read_excel(configs.DATA_PATH,sheet_name=configs.SHEET_NAME)  # 加载数据
    # data["原创微博"] = data["原创微博"].apply(lambda x: text_pretreatment(x))  # 去除话题 只取前256个字符
    # for label in configs.LABEL_SETS:  # 遍历所有标签
    #     print(f"{label} -> {data[label].values.tolist().count(1)}")  # 统计所有标签出现的次数
    # X = data["原创微博"].tolist()  # 输入
    # y = data[configs.LABEL_SETS].values.tolist()  # 输出
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.12, random_state=42)  # 按照7:1:2切分数据集
    # save_data(X_train, X_test, y_train, y_test)  # 保存数据
    X_train_loaded, X_test_loaded, y_train_loaded, y_test_loaded = load_saved_data()
    return X_train_loaded, X_test_loaded, y_train_loaded, y_test_loaded
    # return X_train_loaded[400:500], X_test_loaded[400:500], y_train_loaded[400:500], y_test_loaded[400:500]


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()
