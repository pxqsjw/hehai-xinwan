#!/usr/bin/env python
# -*- coding: utf-8 -*-


import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report
import torch.nn.functional as F

import torch
from transformers import AutoTokenizer

from emotion_opt import configs
from model import EmotionBertModel
from emotion_pretreatment import text_pretreatment
import numpy as np

TOKENIZER = AutoTokenizer.from_pretrained(configs.BERT_DIR)  # bert分词器
MODEL = EmotionBertModel(configs.BERT_DIR, outputs_num=len(configs.LABEL_SETS))  # 模型
MODEL.load_state_dict(torch.load(configs.BERT_MODEL_PATH, map_location=torch.device("cpu")))  # 加载模型参数
MODEL.to(configs.DEVICE)
MODEL.eval()  # 验证模式
print("previous model loading...")


def to_inputs(batch_texts):
    batch_token_ripe = TOKENIZER.batch_encode_plus(
        batch_texts,
        padding=True,
        return_offsets_mapping=True,
    )  # bert分词 padding到该batch的最大长度
    return (
        torch.LongTensor(batch_token_ripe["input_ids"]).to(configs.DEVICE),
        torch.ByteTensor(batch_token_ripe["attention_mask"]).to(configs.DEVICE),
    )


def predict(batch_text):
    batch_texts = [text_pretreatment(text) for text in batch_text]  # 预处理
    batch_inputs, batch_mask = to_inputs(batch_texts)  # to token
    with torch.no_grad():
        logits = MODEL(batch_inputs, attention_mask=batch_mask) # 预测
        # 使用 softmax 将 logits 转换为概率
        probs = F.softmax(logits, dim=1)  # softmax 操作
        results = torch.argmax(probs, dim=1).float().cpu().numpy().astype(int).tolist()
        # 四舍五入到指定的小数位
        probs_rounded = np.round(probs.cpu().numpy(), decimals=3).tolist()
    return results, probs_rounded


if __name__ == "__main__":
    # 创建反向字典，值作为键，键作为值
    reverse_label_dict = {v: k for k, v in configs.LABEL_DICT.items()}

    # 预测单句子
    text = """#昨晚的微博评论太好哭了##1分钟看郑州暴雨救援感人瞬间# 应该让我们的孩子，在一次次社会课堂中，感受那些美好的中国精神，倾听那些颇具力量的声音，敬仰那些勇敢奔赴的身影，悲悯那些美好生命的逝去。这个社会、国家、世界，终究属于他们，他们需要用这些最好的东西武装成长，眼睛里流露的应是想缔造更好未来的星星✨希望我们的家长加以引导，抓住一些教育时机，也是增加亲子沟通的机会，暑假生活有见有闻，有行有思，潜移默化地影响咱们的孩子（来自班主任的日常唠叨） 宜昌 """
    results, socres = predict([text])
    print("预测标签为:", reverse_label_dict[results[0]])

    # 预测文件
    predict_f  = configs.TEST_DATA_PATH
    predict_data = pd.read_excel(predict_f)[:]
    all_text = predict_data["sentence"].tolist()

    batch_size = 32
    batch_num = len(all_text) // batch_size + 1
    results, logits = [], []
    for i in tqdm(range(batch_num), desc="正在预测", total=batch_num):
        batch_text = all_text[i * batch_size : (i + 1) * batch_size]
        result, logit  = predict(batch_text)
        results.extend(result)
        logits.extend(logit)


    predict_data["预测标签"] = results
    predict_data["预测得分"] = logits

    # 拆分 emotion_values 列为多个列并命名
    predict_data[configs.LABEL_SETS] = pd.DataFrame(predict_data['预测得分'].tolist(), index=predict_data.index)
    predict_data.drop(columns=['预测得分'], inplace=True)
    # predict_data = predict_data.rename(columns={'X': 'Text', 'y': '原始标签'})
    predict_data["预测标签"] = [reverse_label_dict[num] for num in results]
    # predict_data["sentiment"] = [reverse_label_dict[num] for num in  predict_data["原始标签"]]

    # 计算分类报告
    y_true = predict_data["sentiment"]  # 原始标签
    y_pred = predict_data["预测标签"]  # 预测标签
    # 添加一列 '一致性'，表示标签是否一致
    predict_data["一致性"] = (y_true == y_pred)
    unique_labels = sorted(list(set(y_true) | set(y_pred)))
    report = classification_report(y_true, y_pred,
                                   target_names=[str(label) for label in unique_labels],
                                   labels=unique_labels,
                                   digits=3)

    # 打印报告到控制台
    print(report)

    # 保存分类报告到 Markdown 文件
    report_filename = f"{configs.OUTPUTS_DIR}/{configs.task}_classification_report.md"

    with open(report_filename, 'w') as f:
        f.write("# 分类报告\n\n")
        f.write("以下是模型的分类评估报告：\n\n")
        f.write("```text\n")
        f.write(report)
        f.write("\n```")
    predict_data.to_excel(f"{configs.OUTPUTS_DIR}/{configs.task}_predict_result.xlsx", index=False)
