import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import precision_recall_fscore_support
from torch.optim import Adam

# 数据加载和预处理
import pandas as pd

# 加载数据
data = pd.read_excel('D:/hehai/LSTM/训练数据.xlsx')  # 替换为您的文件路径

# 提取文本和标签
texts = data['原创微博'].values
labels = data.drop(columns=['原创微博']).values

# 转换标签为二进制格式（多标签分类）
mlb = MultiLabelBinarizer()
labels_bin = mlb.fit_transform(labels)

# 划分数据集为训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(texts, labels_bin, test_size=0.2, random_state=42)


# 自定义数据集类
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.float)
        encoding = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_len,
                                  return_tensors='pt')
        input_ids = encoding['input_ids'].squeeze(0)  # 去除批次维度
        attention_mask = encoding['attention_mask'].squeeze(0)  # 去除批次维度
        return input_ids, attention_mask, label


# 定义tokenizer和最大序列长度
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
max_len = 128

# 创建训练和验证数据集
train_dataset = TextDataset(X_train, y_train, tokenizer, max_len)
val_dataset = TextDataset(X_val, y_val, tokenizer, max_len)

# 创建DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)


# 定义LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, input_ids, attention_mask):
        lstm_out, _ = self.lstm(input_ids)
        output = self.fc(lstm_out[:, -1, :])  # 只使用最后一个时间步的输出
        return output


# 初始化模型、损失函数和优化器
input_dim = 768  # BERT的嵌入维度
hidden_dim = 128  # LSTM的隐藏层维度
output_dim = labels_bin.shape[1]  # 标签集的维度

model = LSTMModel(input_dim, hidden_dim, output_dim)
model.to('cuda' if torch.cuda.is_available() else 'cpu')  # 将模型移到GPU或CPU

# 损失函数和优化器
criterion = nn.BCEWithLogitsLoss()  # 二元交叉熵损失函数
optimizer = Adam(model.parameters(), lr=3e-5)


# 训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0
        correct_preds = 0
        total_preds = 0
        total_labels = 0

        for input_ids, attention_mask, labels in train_loader:
            input_ids = input_ids.to('cuda' if torch.cuda.is_available() else 'cpu')
            attention_mask = attention_mask.to('cuda' if torch.cuda.is_available() else 'cpu')
            labels = labels.to('cuda' if torch.cuda.is_available() else 'cpu')

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # 计算每个batch的precision, recall, f1-score
            preds = torch.sigmoid(outputs).detach().cpu().numpy()
            preds = (preds > 0.5).astype(int)

            precision, recall, f1, support = precision_recall_fscore_support(labels.cpu().numpy(), preds,
                                                                             average='binary', zero_division=1)
            correct_preds += precision.sum()
            total_preds += recall.sum()
            total_labels += support.sum()

        # 每个epoch的平均损失和指标
        avg_loss = total_loss / len(train_loader)
        avg_precision = correct_preds / total_preds if total_preds > 0 else 0
        avg_recall = correct_preds / total_labels if total_labels > 0 else 0
        avg_f1 = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) if (
                                                                                                avg_precision + avg_recall) > 0 else 0

        print(
            f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}, Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}, F1-Score: {avg_f1:.4f}")

    return model


# 训练模型
trained_model = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=3)  # 训练3个epoch
