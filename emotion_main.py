#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/2/7 20:53
@Author  : Callion.lin
@File    : emotion_main.py
@Description:  
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AdamW, get_cosine_schedule_with_warmup

from emotion_opt import configs
from data_loader import Datasets
from model import EmotionBertModel
from emotion_pretreatment import load_data
from emotion_train import train


def get_label_weights(labels):
    # 计算类别权重
    counts = np.array(labels).sum(axis=0).tolist()
    label_ratios = dict(zip(configs.LABEL_SETS, [round(count / sum(counts), 4) for count in counts]))
    print(f"label ratios: {label_ratios}")
    weights = [sum(counts) / count for count in counts]
    weights = [round(weight / sum(weights), 4) for weight in weights]
    label_weights = dict(zip(configs.LABEL_SETS, weights))
    print(f"label weights: {label_weights}")
    return weights


def train_run():
    print(f"current device: {configs.DEVICE}")

    X_train, X_val, _,y_train, y_val,_ = load_data()  # 加载数据
    print(f"train data length: {len(X_train)}, val data length: {len(X_val)}")

    train_dataset = Datasets(X_train, y_train, configs.BERT_DIR, configs.DEVICE)  # 训练数据集
    val_dataset = Datasets(X_val, y_val, configs.BERT_DIR, configs.DEVICE)  # 验证数据集
    print("dataset build...")

    train_loader = DataLoader(
        train_dataset,
        batch_size=configs.BATCH_SIZE,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
    )  # 训练数据加载器
    val_loader = DataLoader(
        val_dataset,
        batch_size=configs.BATCH_SIZE,
        shuffle=True,
        collate_fn=val_dataset.collate_fn,
    )  # 验证数据加载器
    print("get dataLoader...")

    model = EmotionBertModel(configs.BERT_DIR, outputs_num=len(configs.LABEL_SETS)).to(configs.DEVICE)  # 定义模型

    bert_optimizer = list(model.bert.named_parameters())  # bert参数
    classifier_optimizer = list(model.classifier.named_parameters())  # classifier参数
    if configs.FINE_TUNING:  # 微调bert
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]  # 权重衰减中不衰减元素
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in bert_optimizer if not any(nd in n for nd in no_decay)],
                "weight_decay": configs.WEIGHT_DECAY,
            },
            {
                "params": [p for n, p in bert_optimizer if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
            {
                "params": [p for n, p in classifier_optimizer if not any(nd in n for nd in no_decay)],
                "lr": configs.CLASSIFIER_LEARNING_RATE,
                "weight_decay": configs.WEIGHT_DECAY,
            },
            {
                "params": [p for n, p in classifier_optimizer if any(nd in n for nd in no_decay)],
                "lr": configs.CLASSIFIER_LEARNING_RATE,
                "weight_decay": 0.0,
            },
        ]
    else:  # 固定bert，只调整其他层
        optimizer_grouped_parameters = [{"params": [p for n, p in classifier_optimizer]}]
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=configs.LEARNING_RATE,
        correct_bias=False,
        no_deprecation_warning=True,
    )  # 优化器

    # criterion = torch.nn.BCEWithLogitsLoss(
    #     weight=torch.FloatTensor(get_label_weights(y_train)).to(configs.DEVICE)
    # )  # 损失函数
    criterion = torch.nn.CrossEntropyLoss()  # 损失函数

    train_steps_per_epoch = len(train_dataset) // configs.BATCH_SIZE
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=(configs.EPOCH_NUM // 10) * train_steps_per_epoch,
        num_training_steps=configs.EPOCH_NUM * train_steps_per_epoch,
    )  # warm_up策略 学习率衰减

    print("start training...")
    train(train_loader, val_loader, model, optimizer, criterion, scheduler, configs.BERT_MODEL_PATH)  # 开始训练


if __name__ == "__main__":
    train_run()  # 训练模型
