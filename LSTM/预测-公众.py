import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report
import pandas as pd

# 设置 pandas 显示选项，确保所有浮点数显示为三位小数
pd.set_option('display.float_format', '{:.3f}'.format)

# 加载保存的 LSTM 模型和分词器
model = load_model('lstm_model_best.keras')

with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# 加载测试数据集
test_file_path = 'D:/hehai/LSTM/公众测试.xlsx'  # 更新路径
test_df = pd.read_excel(test_file_path)

# 准备文本数据用于预测
X_test = test_df['原创微博'].values

# 对文本进行分词和填充
X_seq = tokenizer.texts_to_sequences(X_test)
X_pad = pad_sequences(X_seq, maxlen=100)

# 预测标签
predictions = model.predict(X_pad)
predicted_labels = (predictions > 0.5).astype(int)  # 使用 0.5 作为阈值进行多标签分类

# 将预测标签更新到数据框的 "预测标签" 列
test_df['预测标签'] = predicted_labels.tolist()

# 计算一致性评分（正确预测的标签数量）
correct_predictions = np.sum(predicted_labels == test_df[['Problem solving', 'Instrumental social support offering', 'Emotional social support offering',
        'Escape and vent', 'Egoism and Criminality']].values, axis=1)
test_df['一致性评分'] = correct_predictions

# 显示分类报告
y_true = test_df[['Problem solving', 'Instrumental social support offering', 'Emotional social support offering',
        'Escape and vent', 'Egoism and Criminality']].values
classification_rep = classification_report(y_true, predicted_labels, target_names=[
    'Problem solving', 'Instrumental social support offering', 'Emotional social support offering',
        'Escape and vent', 'Egoism and Criminality'], output_dict=False)

# 打印分类报告
print("Classification Report:\n", classification_rep)

# 可选：将更新后的数据框保存到新的 Excel 文件
test_df.to_excel('predicted_output.xlsx', index=False)

# 将分类报告转换为字典格式
classification_report_dict = classification_report(
    y_true,
    predicted_labels,
    target_names=[
        'Problem solving', 'Instrumental social support offering', 'Emotional social support offering',
        'Escape and vent', 'Egoism and Criminality'
    ],
    output_dict=True
)

# 格式化分类报告中的数值，确保每个数字为三位小数
formatted_metrics = {
    "precision": [round(classification_report_dict[label]['precision'], 3) for label in classification_report_dict if label != 'accuracy'],
    "recall": [round(classification_report_dict[label]['recall'], 3) for label in classification_report_dict if label != 'accuracy'],
    "f1-score": [round(classification_report_dict[label]['f1-score'], 3) for label in classification_report_dict if label != 'accuracy'],
    "support": [classification_report_dict[label]['support'] for label in classification_report_dict if label != 'accuracy']  # 'support' 无需四舍五入
}

# 将这些格式化后的值放到一个 DataFrame 中
formatted_df = pd.DataFrame(formatted_metrics, index=[label for label in classification_report_dict if label != 'accuracy'])

# 打印格式化后的分类报告
print("Formatted Classification Report:")
print(formatted_df.to_string(index=True, float_format="{:.3f}".format))
