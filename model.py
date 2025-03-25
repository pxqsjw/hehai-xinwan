from torch import nn
from transformers import AutoConfig, AutoModel
import torch

class BertModel(nn.Module):
    def __init__(self, bert_path, outputs_num):
        super(BertModel, self).__init__()

        self.bert_config = AutoConfig.from_pretrained(bert_path)  # bert配置
        self.bert = AutoModel.from_pretrained(bert_path)  # bert模型
        self.classifier = nn.Linear(self.bert_config.hidden_size, outputs_num)  # 分类器
        self.dropout = nn.Dropout(0.2)  # dropout

    def forward(self, input_ids, attention_mask):
        # print("1", input_ids.size())  # torch.Size([B, F])
        # print("2", attention_mask.size())  # torch.Size([B, F])

        outputs = self.bert(input_ids, attention_mask=attention_mask).last_hidden_state  # [0] = .last_hidden_state
        # print("3", outputs.size())  # torch.Size([B, F, H])

        outputs = self.dropout(outputs[:, 0, :])
        # print("4", outputs.size())  # torch.Size([B, H])

        outputs = self.classifier(outputs)
        # print("5", outputs.size())  # torch.Size([B, O])

        return outputs


class EmotionBertModel(nn.Module):
    def __init__(self, bert_path, outputs_num):
        """
        初始化EmotionBertModel模型。

        Args:
            bert_path (str): 预训练BERT模型的路径。
            outputs_num (int): 分类任务的输出类别数。
        """
        super(EmotionBertModel, self).__init__()

        # BERT配置和模型
        self.bert_config = AutoConfig.from_pretrained(bert_path)
        self.bert = AutoModel.from_pretrained(bert_path)

        # 分类器
        self.classifier = nn.Linear(self.bert_config.hidden_size, outputs_num)

        # Dropout层，防止过拟合
        self.dropout = nn.Dropout(0.2)

    def forward(self, input_ids, attention_mask):
        """
        前向传播。

        Args:
            input_ids (Tensor): 输入的token IDs。
            attention_mask (Tensor): 输入的attention mask，用于区分padding部分。

        Returns:
            logits (Tensor): 未经过softmax的模型输出logits，用于计算损失。
            preds (Tensor): 预测的类别索引。
        """
        # 获取BERT的输出
        outputs = self.bert(input_ids, attention_mask=attention_mask).last_hidden_state

        # 只取[CLS] token的输出
        cls_output = outputs[:, 0, :]

        # Dropout层
        cls_output = self.dropout(cls_output)

        # 分类器层
        logits = self.classifier(cls_output)

        # 返回logits和preds
        return logits