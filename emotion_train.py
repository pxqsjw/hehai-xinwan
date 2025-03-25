#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/2/7 22:55
@Author  : Callion.lin
@File    : emotion_train.py
@Description:  
"""

import os
import sys

import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch import nn
from tqdm import tqdm

from emotion_opt import  configs
from emotion_utils import epoch_visualization, save_evaluate



# def train_epoch(train_loader, model, optimizer, criterion, scheduler, epoch):
#     model.train()  # 训练模式
#     real_targets = []  # 真实标签
#     pred_targets = []  # 预测标签
#     train_loss_records = []  # loss
#
#     for idx, batch_samples in enumerate(tqdm(train_loader, file=sys.stdout)):
#         batch_inputs, batch_mask, batch_targets, batch_texts = batch_samples
#
#         outputs = model(batch_inputs, attention_mask=batch_mask)  # 前向传播
#         loss = criterion(outputs, batch_targets)  # 计算loss
#         optimizer.zero_grad()  # 梯度清零
#         loss.backward()  # 反向传播
#         nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=configs.CLIP_GRAD)  # 梯度裁剪
#         optimizer.step()  # 参数更新
#         scheduler.step()  # 学习率衰减
#
#         real_targets.extend(batch_targets.cpu().numpy().astype(int).tolist())  # 记录真实标签
#         pred_targets.extend((torch.sigmoid(outputs) > 0.5).float().cpu().numpy().astype(int).tolist())  # 记录预测标签
#         train_loss_records.append(loss.item())  # 记录loss
#
#     train_f1 = f1_score(real_targets, pred_targets, average="micro", zero_division=0)  # 计算f1
#     train_loss = sum(train_loss_records) / len(train_loss_records)  # 求平均
#     print(f"[train] Epoch: {epoch} / {configs.EPOCH_NUM}, f1: {train_f1:.4f}, loss: {train_loss:.4f}")
#
#     return train_f1, train_loss
def train_epoch(train_loader, model, optimizer, criterion, scheduler, epoch):
    model.train()  # 训练模式
    real_targets = []  # 真实标签
    pred_targets = []  # 预测标签
    train_loss_records = []  # loss

    for idx, batch_samples in enumerate(tqdm(train_loader, file=sys.stdout)):
        batch_inputs, batch_mask, batch_targets, batch_texts = batch_samples
        batch_targets = batch_targets.long()
        logits = model(batch_inputs, attention_mask=batch_mask)  # 前向传播
        loss = criterion(logits, batch_targets)  # 计算loss
        optimizer.zero_grad()  # 梯度清零
        loss.backward()  # 反向传播
        nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=configs.CLIP_GRAD)  # 梯度裁剪
        optimizer.step()  # 参数更新
        scheduler.step()  # 学习率衰减

        real_targets.extend(batch_targets.cpu().numpy().tolist())  # 记录真实标签
        pred_targets.extend(torch.argmax(logits, dim=1).cpu().numpy().tolist())  # 记录预测标签
        train_loss_records.append(loss.item())  # 记录loss

    train_f1 = f1_score(real_targets, pred_targets, average="micro", zero_division=0)  # 计算f1
    train_loss = sum(train_loss_records) / len(train_loss_records)  # 求平均
    print(f"[train] Epoch: {epoch} / {configs.EPOCH_NUM}, f1: {train_f1:.4f}, loss: {train_loss:.4f}")

    return train_f1, train_loss

# def evaluate(loader, model, criterion, epoch, mode="val"):
#     model.eval()  # 验证模式
#     real_targets = []  # 真实标签
#     pred_targets = []  # 预测标签
#     val_loss_records = []  # loss
#
#     for idx, batch_samples in enumerate(loader) if mode == "val" else enumerate(tqdm(loader, file=sys.stdout)):
#         batch_inputs, batch_mask, batch_targets, batch_texts = batch_samples
#
#         with torch.no_grad():
#             outputs = model(batch_inputs, attention_mask=batch_mask)  # 预测
#             loss = criterion(outputs, batch_targets)  # 计算loss
#
#         real_targets.extend(batch_targets.cpu().numpy().astype(int).tolist())  # 记录真实标签
#         pred_targets.extend((torch.sigmoid(outputs) > 0.5).float().cpu().numpy().astype(int).tolist())  # 记录预测标签
#         val_loss_records.append(loss.item())  # 记录loss
#
#     val_f1 = f1_score(real_targets, pred_targets, average="micro", zero_division=0)  # 计算f1
#     val_loss = sum(val_loss_records) / len(val_loss_records)  # 求平均
#     print(f"[val]   Epoch: {epoch} / {configs.EPOCH_NUM}, f1: {val_f1:.4f}, loss: {val_loss:.4f}")
#     return val_f1, val_loss, real_targets, pred_targets
def evaluate(loader, model, criterion, epoch, mode="val"):
    model.eval()  # 验证模式
    real_targets = []  # 真实标签
    pred_targets = []  # 预测标签
    val_loss_records = []  # loss

    for idx, batch_samples in enumerate(loader) if mode == "val" else enumerate(tqdm(loader, file=sys.stdout)):
        batch_inputs, batch_mask, batch_targets, batch_texts = batch_samples

        with torch.no_grad():
            batch_targets = batch_targets.long()
            outputs = model(batch_inputs, attention_mask=batch_mask)  # 预测
            loss = criterion(outputs, batch_targets)  # 计算loss

        real_targets.extend(batch_targets.cpu().numpy().tolist())  # 记录真实标签
        pred_targets.extend(torch.argmax(outputs, dim=1).cpu().numpy().tolist())  # 记录预测标签
        val_loss_records.append(loss.item())  # 记录loss

    val_f1 = f1_score(real_targets, pred_targets, average="micro", zero_division=0)  # 计算f1
    val_loss = sum(val_loss_records) / len(val_loss_records)  # 求平均
    print(f"[{mode}] Epoch: {epoch} / {configs.EPOCH_NUM}, f1: {val_f1:.4f}, loss: {val_loss:.4f}")
    return val_f1, val_loss, real_targets, pred_targets


def train(train_loader, dev_loader, model, optimizer, criterion, scheduler, model_path):
    best_val_f1 = 0.0  # 最佳f1
    patience_counter = 0  # 耐心值
    train_f1_records = []  # 训练f1
    train_loss_records = []  # 训练loss
    val_f1_records = []  # 验证f1
    val_loss_records = []  # 验证loss

    for epoch in range(1, configs.EPOCH_NUM + 1):
        train_f1, train_loss = train_epoch(train_loader, model, optimizer, criterion, scheduler, epoch)  # 训练
        val_f1, val_loss, real_targets, pred_targets = evaluate(dev_loader, model, criterion, epoch, mode="val")  # 验证

        train_f1_records.append(train_f1)  # 记录
        train_loss_records.append(train_loss)  # 记录
        val_f1_records.append(val_f1)  # 记录
        val_loss_records.append(val_loss)  # 记录

        if val_f1 - best_val_f1 > configs.PATIENCE:  # 如果比之前模型效果好
            best_val_f1 = val_f1  # 记录最佳acc
            torch.save(
                model.state_dict(),
                model_path
                if configs.IS_COVER
                else os.path.join(
                    configs.OUTPUTS_DIR,
                    f"{epoch}-train_f1{train_f1:.4f}-val_f1{val_f1:.4f}-model.pkl",
                ),
            )  # 存储模型
            save_evaluate(real_targets, pred_targets, configs.RESULT_PATH)  # 存储模型指标
            print("--------Save the best model--------")
            patience_counter = 0
        else:
            patience_counter += 1

        if (patience_counter >= configs.PATIENCE_NUM and epoch > configs.MIN_EPOCH_NUM) or epoch == configs.EPOCH_NUM:
            print(f"Terminate training in advance, best F1: {best_val_f1:.4f}")
            break

    epoch_visualization(train_loss_records, val_loss_records, "Loss", configs.LOSS_VISUALIZATION_PATH)  # 绘制loss图
    pd.DataFrame(
        {
            "epoch": list(range(1, len(train_loss_records) + 1)),
            "train loss": train_loss_records,
            "val loss": val_loss_records,
        }
    ).to_csv(
        configs.LOSS_VISUALIZATION_CSV_PATH, index=False
    )  # 保存loss数据
    epoch_visualization(train_f1_records, val_f1_records, "F1 Score", configs.F1_VISUALIZATION_PATH)  # 绘制f1图
    pd.DataFrame(
        {
            "epoch": list(range(1, len(train_f1_records) + 1)),
            "train f1 score": train_f1_records,
            "val f1 score": val_f1_records,
        }
    ).to_csv(
        configs.F1_VISUALIZATION_CSV_PATH, index=False
    )  # 保存loss数据
    print("--------Model training ends--------")
