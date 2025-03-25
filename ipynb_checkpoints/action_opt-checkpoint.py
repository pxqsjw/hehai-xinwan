import os
import numpy as np
import random
import torch


class Opt:
    def __init__(self, seed=42):
        self.ROOT_DIR = os.path.abspath(os.path.dirname(__file__))

        # 数据
        self.DATA_DIR = os.path.join(self.ROOT_DIR, "data")
        self.DATA_PATH = os.path.join(self.DATA_DIR, "标注数据.xlsx")
        self.PREDICT_DATA_PATH = os.path.join(self.DATA_DIR, "20250215-行为数据测试集.xlsx")
        # self.PREDICT_RESULT_PATH = os.path.join(self.DATA_DIR, "行为数据3770预测数据结果.xlsx")
        self.SHEET_NAME = "标注公众 (新)"  # 用哪个数据   标注政府 (新) 标注公众 (新)

        # 模型输出
        self.OUTPUTS_DIR = os.path.join(self.ROOT_DIR, f"outputs_{self.SHEET_NAME}")
        self.BERT_DIR = os.path.join("", "albert_chinese_tiny")
        self.BERT_MODEL_PATH = os.path.join(self.OUTPUTS_DIR, f"{self.BERT_DIR}_model.pkl")
        self.LOSS_VISUALIZATION_PATH = os.path.join(self.OUTPUTS_DIR, f"{self.BERT_DIR}_loss_visualization.png")
        self.LOSS_VISUALIZATION_CSV_PATH = os.path.join(self.OUTPUTS_DIR, f"{self.BERT_DIR}_loss_visualization.csv")
        self.F1_VISUALIZATION_PATH = os.path.join(self.OUTPUTS_DIR, f"{self.BERT_DIR}_f1_visualization.png")
        self.F1_VISUALIZATION_CSV_PATH = os.path.join(self.OUTPUTS_DIR, f"{self.BERT_DIR}_f1_visualization.csv")
        self.RESULT_PATH = os.path.join(self.OUTPUTS_DIR, f"{self.BERT_DIR}_result.txt")
        self.TRAIN_DATA_PATH = f"{self.DATA_DIR}/train_{self.SHEET_NAME}.xlsx"
        self.VAL_DATA_PATH = f"{self.DATA_DIR}/val_{self.SHEET_NAME}.xlsx"

        # 标签集
        if self.SHEET_NAME == "标注公众 (新)":
            self.LABEL_SETS = [
                "Problem solving",
                "Instrumental social support offering",
                "Emotional social support offering",
                "Escape and vent",
                "Egoism and Criminality"
            ]
        elif  self.SHEET_NAME == "标注政府 (新)":
            self.LABEL_SETS = [
            # "Problem solving",
            # "Instrumental social support offering",
            # "Emotional social support offering",
            # "Escape and vent",
            # "Egoism and Criminality",
            #
            "Command and coordination" ,
            "Personnel rescue and relocation" ,
            "Facility defense and repairs",
            "Information release and crisis communication" ,
            "Recovery of economy and life"
        ]

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

configs = Opt(seed=42)  # 初始化 Opt 类并设置随机种子

