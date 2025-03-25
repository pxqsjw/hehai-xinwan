import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support

# 数据加载
data = pd.read_excel('D:/hehai/LSTM/训练数据.xlsx')

# 提取文本和标签
texts = data['原创微博'].values  # 输入文本数据
labels = data.drop(columns=['原创微博']).values  # 其余为标签列（多标签）

# 转换标签为二进制格式（多标签分类）
mlb = MultiLabelBinarizer()
labels_bin = mlb.fit_transform(labels)

# 划分数据集为训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(texts, labels_bin, test_size=0.2, random_state=42)


# 自定义数据集类
class TextDataset(Dataset):
    def __init__(self, texts, labels, word_index, max_len):
        self.texts = texts
        self.labels = labels
        self.word_index = word_index
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.float)
        sequence = self.text_to_sequence(text)
        return torch.tensor(sequence, dtype=torch.long), label

    def text_to_sequence(self, text):
        words = text.split()  # 按空格分词
        sequence = [self.word_index.get(word, 0) for word in words]  # 获取每个词的索引，若词不在词典中则用0填充
        if len(sequence) < self.max_len:
            sequence.extend([0] * (self.max_len - len(sequence)))  # 填充至max_len
        else:
            sequence = sequence[:self.max_len]  # 截断至max_len
        return sequence


# 构建词汇表
word_index = {}
word_index['<PAD>'] = 0  # 为了进行填充，设置PAD标记
index = 1
for text in texts:
    for word in text.split():
        if word not in word_index:
            word_index[word] = index
            index += 1


# LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_layers=1):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)  # 嵌入层
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)  # 获取文本的词向量
        lstm_out, (hn, cn) = self.lstm(x)  # LSTM层的输出
        out = self.fc(lstm_out[:, -1, :])  # 使用最后一个时间步的输出
        return out


# 数据准备
max_len = 128  # 设置最大文本长度
embedding_dim = 100  # 嵌入层的维度
hidden_dim = 128
output_dim = labels_bin.shape[1]  # 类别数量

# 创建训练和验证数据集
train_dataset = TextDataset(X_train, y_train, word_index, max_len)
val_dataset = TextDataset(X_val, y_val, word_index, max_len)

# 创建DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# 初始化模型、损失函数和优化器
vocab_size = len(word_index)
model = LSTMModel(vocab_size, embedding_dim, hidden_dim, output_dim)
model.to('cuda' if torch.cuda.is_available() else 'cpu')

# 损失函数和优化器
criterion = nn.BCEWithLogitsLoss()  # 二分类交叉熵损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)


# 训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    train_loss_history = []
    train_precision_history = []
    train_recall_history = []
    train_f1_history = []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        epoch_preds = []
        epoch_labels = []

        for inputs, labels in train_loader:
            optimizer.zero_grad()

            # 前向传播
            outputs = model(inputs)

            # 计算损失
            loss = criterion(outputs, labels.float())  # BCEWithLogitsLoss要求标签是浮点型
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # 获取预测结果
            preds = torch.sigmoid(outputs).round()  # 输出0或1
            epoch_preds.append(preds)
            epoch_labels.append(labels)

        # 计算精确度、召回率和F1得分
        epoch_preds = torch.cat(epoch_preds, dim=0).detach().cpu().numpy()
        epoch_labels = torch.cat(epoch_labels, dim=0).detach().cpu().numpy()

        precision, recall, f1, _ = precision_recall_fscore_support(epoch_labels, epoch_preds, average='macro')

        # 记录训练过程中的数据
        train_loss_history.append(epoch_loss / len(train_loader))
        train_precision_history.append(precision)
        train_recall_history.append(recall)
        train_f1_history.append(f1)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / len(train_loader):.4f}, "
              f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

    return train_loss_history, train_precision_history, train_recall_history, train_f1_history


# 训练并可视化
train_loss_history, train_precision_history, train_recall_history, train_f1_history = train_model(
    model, train_loader, val_loader, criterion, optimizer, num_epochs=10
)

# 绘制训练过程的图形
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(train_loss_history, label="Training Loss")
plt.title("Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")

plt.subplot(2, 2, 2)
plt.plot(train_precision_history, label="Training Precision")
plt.title("Training Precision")
plt.xlabel("Epochs")
plt.ylabel("Precision")

plt.subplot(2, 2, 3)
plt.plot(train_recall_history, label="Training Recall")
plt.title("Training Recall")
plt.xlabel("Epochs")
plt.ylabel("Recall")

plt.subplot(2, 2, 4)
plt.plot(train_f1_history, label="Training F1 Score")
plt.title("Training F1 Score")
plt.xlabel("Epochs")
plt.ylabel("F1 Score")

plt.tight_layout()
plt.show()
# 保存模型
torch.save(model.state_dict(), 'lstm_model.pth')  # 保存为 lstm_model.pth
