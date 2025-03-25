#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/2/7 20:55
@Author  : Callion.lin
@File    : emotion_opt.py
@Description:  
"""

import os
import numpy as np
import random
import torch
import pandas as  pd


class Opt:
    def __init__(self, seed=42):
        self.ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
        self.use_context = False
        self.max_words = 5

        # 数据
        self.task = "emotion"
        self.DATA_DIR = os.path.join(self.ROOT_DIR, "data")
        self.DATA_PATH = os.path.join(self.DATA_DIR, "情感标注汇总14049条.xlsx")
        self.TRAIN_DATA_PATH = f"{self.DATA_DIR}/train_emotion.xlsx"
        self.VAL_DATA_PATH = f"{self.DATA_DIR}/val_emotion.xlsx"
        self.TEST_DATA_PATH = f"{self.DATA_DIR}/test_emotion.xlsx"
        self.WORD_DICT_PATH = os.path.join(self.DATA_DIR, "大连理工大学中文情感词汇本体（补充）最新版2.xlsx")
        self.DEGREE_DICT_PATH = os.path.join(self.DATA_DIR, "degree_dict.txt")
        self.NO_WORD_PATH = os.path.join(self.DATA_DIR, "no.txt")



        # 模型输出
        self.OUTPUTS_DIR = os.path.join(self.ROOT_DIR, f"{self.task}_outputs")
        self.BERT_DIR = os.path.join("", "albert_chinese_tiny") # albert_chinese_tiny
        self.BERT_MODEL_PATH = os.path.join(self.OUTPUTS_DIR, f"{self.task}_model.pkl")
        self.LOSS_VISUALIZATION_PATH = os.path.join(self.OUTPUTS_DIR, f"{self.task}_loss_visualization.png")
        self.LOSS_VISUALIZATION_CSV_PATH = os.path.join(self.OUTPUTS_DIR, f"{self.task}_loss_visualization.csv")
        self.F1_VISUALIZATION_PATH = os.path.join(self.OUTPUTS_DIR, f"{self.task}_f1_visualization.png")
        self.F1_VISUALIZATION_CSV_PATH = os.path.join(self.OUTPUTS_DIR, f"{self.task}_f1_visualization.csv")
        self.RESULT_PATH = os.path.join(self.OUTPUTS_DIR, f"{self.task}_result.txt")

        # self.word_dict, self.degree_dict, self.no_word_dict = self.load_dictionaries()


        # 标签集
        self.LABEL_SETS = [
            "positive",
            "anger",
            "fear",
            "sadness"
        ]
        self.LABEL_DICT = {
            "positive": 0,
            "anger": 1,
            "fear": 2,
            "sadness": 3
        }
        # 设备
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 模型加载
        self.IS_COVER = True  # 是否在训练阶段覆盖模型
        self.FINE_TUNING = True  # 是否对整个BERT进行fine tuning

        # 设置随机种子
        self.setup_seed(seed)

        # 模型参数
        self.LEARNING_RATE = 3e-5  # 基础学习率
        self.CLASSIFIER_LEARNING_RATE = self.LEARNING_RATE * 100  # classifier学习率
        self.WEIGHT_DECAY = 0.01  # 权重衰减因子
        self.CLIP_GRAD = 5  # 梯度裁剪因子
        self.BATCH_SIZE = 32  # batch
        self.EPOCH_NUM = 200  # epoch
        self.MIN_EPOCH_NUM = 5  # 最小epoch
        self.PATIENCE = 0.0002  # 每次必提高指标
        self.PATIENCE_NUM = 10  # 耐心度
        self.makedir(self.OUTPUTS_DIR)  # 使用 makedir 方法创建文件夹

    def makedir(self, _dir):
        # 新建文件夹
        if not os.path.exists(_dir):
            os.makedirs(_dir)

    def setup_seed(self, seed=42):
        # 设置随机种子 保证模型的可复现性
        os.environ["PYTHONHASHSEED"] = str(seed)

        np.random.seed(seed)
        random.seed(seed)

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False

    # 读取WORD_DICT_PATH文件，构建情感字典
    def load_word_dict(self, word_dict_path):
            word_df = pd.read_excel(word_dict_path)  # 读取Excel文件
            word_dict = dict(zip(word_df["词语"], word_df["情感类别"]))  # 形成字典
            # print(word_dict)
            return word_dict

    # 读取DEGREE_DICT_PATH文件，构建程度副词字典
    def load_degree_dict(self, degree_dict_path):
            degree_dict = {}
            with open(degree_dict_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()  # 以空格分割，得到词和程度
                    if len(parts) == 2:
                        word, degree = parts
                        degree_dict[word] ="程度副词"  # 存储为字典，词与程度
            # print(degree_dict)
            return degree_dict

    # 读取NO_WORD_PATH文件，构建否定词字典
    def load_no_word_dict(self, no_word_path):
            no_word_dict = {}
            with open(no_word_path, 'r', encoding='utf-8') as f:
                for line in f:
                    word = line.strip()
                    no_word_dict[word] = "否定词"  # 否定词字典，值为True表示否定词
            # print(no_word_dict)
            return no_word_dict

    # 调用这些函数来读取和构建字典
    def load_dictionaries(self):
            word_dict = self.load_word_dict(self.WORD_DICT_PATH)  # 情感字典
            degree_dict = self.load_degree_dict(self.DEGREE_DICT_PATH)  # 程度副词字典
            no_word_dict = self.load_no_word_dict(self.NO_WORD_PATH)  # 否定词字典
            return word_dict, degree_dict, no_word_dict
            # 加载字典

configs = Opt(seed=42)  # 初始化 Opt 类并设置随机种子
