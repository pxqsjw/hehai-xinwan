#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/2/7 22:52
@Author  : Callion.lin
@File    : emotion_pretreatment.py.py
@Description:  
"""

import re

import pandas as pd
from sklearn.model_selection import train_test_split

from emotion_opt import configs


class TextProcessor:
    def __init__(self, word_dict, degree_dict, no_word_dict, max_words=3):
        self.word_dict = word_dict
        self.degree_dict = degree_dict
        self.no_word_dict = no_word_dict
        self.max_words = max_words  # 设置拼接的最大词数

    # 根据词典检查文本并生成context
    def generate_context(self, text):
        # text = str(text)
        context = []
        seen_words = set()  # 用于存储已经添加的词汇，确保去重

        # 检查否定词并放在最前面
        for word in self.no_word_dict.keys():
            if word in text and word not in seen_words:
                context.append(f"{word} ")
                seen_words.add(word)  # 添加到集合中
            if len(context) >= self.max_words:  # 如果已拼接n个词，则停止
                break

        # 检查情感词
        if len(context) < self.max_words:
            for word, sentiment in self.word_dict.items():
                if str(word) in text and word not in seen_words:
                    context.append(f"{word}是{sentiment}")
                    seen_words.add(word)
                if len(context) >= self.max_words:  # 如果已拼接n个词，则停止
                    break

        # 检查程度副词
        if len(context) < self.max_words:
            for word, degree in self.degree_dict.items():
                if word in text and word not in seen_words:
                    context.append(f"{word} ")
                    seen_words.add(word)
                if len(context) >= self.max_words:  # 如果已拼接n个词，则停止
                    break

        # 拼接context并返回
        context_text = "，".join(context)
        return context_text

    # 对整个DataFrame进行处理
    def process_dataframe(self, df):
        df['context'] = df['text'].map(str).apply(lambda x: self.generate_context(x))
        df['final_text'] = df['text'] + " " + df['context']
        return df


def text_pretreatment(text):
    # text = re.sub("#.*#", "", text)  # 去除话题

    # print(text)
    text = text.lower().strip()  # 去除首尾空格
    text = text[:500]  # 只取前256个字符
    return text


# def save_data(X_train, X_val, X_test, y_train, y_val, y_test):
#     train_data = pd.DataFrame({"X": X_train, "y": [str(y) for y in y_train]})
#     val_data = pd.DataFrame({"X": X_val, "y": [str(y) for y in y_val]})
#     test_data = pd.DataFrame({"X": X_test, "y": [str(y) for y in y_test]})
#
#     train_data.to_excel(configs.TRAIN_DATA_PATH, index=False)
#     val_data.to_excel(configs.VAL_DATA_PATH, index=False)
#     test_data.to_excel(configs.TEST_DATA_PATH, index=False)
#
#
# def load_saved_data():
#     train_data = pd.read_excel(configs.TRAIN_DATA_PATH)
#     val_data = pd.read_excel(configs.VAL_DATA_PATH)
#     test_data = pd.read_excel(configs.TEST_DATA_PATH)
#
#     X_train = train_data["X"].tolist()
#     y_train = train_data["y"].tolist()
#     X_val = val_data["X"].tolist()
#     y_val = val_data["y"].tolist()
#     X_test = test_data["X"].tolist()
#     y_test = test_data["y"].tolist()
#
#     return X_train, X_val, X_test, y_train, y_val, y_test

def save_data(train_data, val_data, test_data):
    """
    保存数据到Excel文件
    :param train_data: 训练集DataFrame
    :param val_data: 验证集DataFrame
    :param test_data: 测试集DataFrame
    """
    # 保存为Excel文件
    train_data.to_excel(configs.TRAIN_DATA_PATH, index=False)
    val_data.to_excel(configs.VAL_DATA_PATH, index=False)
    test_data.to_excel(configs.TEST_DATA_PATH, index=False)


def load_saved_data():
    """
    加载保存的数据
    :return: X_train, X_val, X_test, y_train, y_val, y_test
    """
    # 从Excel文件加载数据
    train_data = pd.read_excel(configs.TRAIN_DATA_PATH)
    val_data = pd.read_excel(configs.VAL_DATA_PATH)
    test_data = pd.read_excel(configs.TEST_DATA_PATH)

    # 提取文本和标签
    X_train = train_data["sentence"].tolist()
    y_train = train_data["sentiment"].map(configs.LABEL_DICT).tolist()  # 映射为数字标签
    X_val = val_data["sentence"].tolist()
    y_val = val_data["sentiment"].map(configs.LABEL_DICT).tolist()
    X_test = test_data["sentence"].tolist()
    y_test = test_data["sentiment"].map(configs.LABEL_DICT).tolist()

    return X_train, X_val, X_test, y_train, y_val, y_test

def load_data():
    """



    # 只需运行一次
    data = pd.read_excel(configs.DATA_PATH)  # 加载数据
    text = "text"
    if configs.use_context:
        word_dict, degree_dict, no_word_dict = configs.load_dictionaries()
        # 初始化TextProcessor
        processor = TextProcessor(word_dict, degree_dict, no_word_dict, configs.max_words)
        # 处理DataFrame
        data = processor.process_dataframe(data)
        text = "final_text"

    data["sentence"] = data[text].apply(lambda x: text_pretreatment(x))  # 去除话题 只取前256个字符
    for label in configs.LABEL_SETS:  # 遍历所有标签
        print(f"{label} -> {data['sentiment'].values.tolist().count(label)}")  # 统计所有标签出现的次数

    # 切分数据集，先按 7:3 切分为训练集和临时集，再将临时集按 1:2 切分为验证集和测试集
    # 使用 stratify 保证每个标签的比例一致
    train_data, temp_data = train_test_split(data, test_size=0.3, random_state=42, stratify=data['sentiment'])
    val_data, test_data = train_test_split(temp_data, test_size=0.66, random_state=42,
                                               stratify=temp_data['sentiment'])

    # 打印每个标签的数量，确保切分后各数据集标签分布正确
    print("Train set:")
    print(train_data['sentiment'].value_counts())
    print("Validation set:")
    print(val_data['sentiment'].value_counts())
    print("Test set:")
    print(test_data['sentiment'].value_counts())


    # 保存切分后的数据
    save_data(train_data, val_data, test_data)
    """
    # 重新加载保存的数据
    X_train_loaded, X_val_loaded, X_test_loaded, y_train_loaded, y_val_loaded, y_test_loaded = load_saved_data()

    return X_val_loaded[:]+ X_test_loaded[:], \
           X_val_loaded[:], X_test_loaded,\
          y_val_loaded[:]+ y_test_loaded[:],\
           y_val_loaded[:], y_test_loaded

if __name__ == "__main__":
    X_train_loaded, X_val_loaded, X_test_loaded, y_train_loaded, y_val_loaded, y_test_loaded = load_data()
