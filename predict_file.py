import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report
from action_opt import configs
from predict import predict

def compute_similarity(y_true, preds):
    similarity = []
    for true, pred in zip(y_true, preds):
        # Count the number of matching labels
        match_count = sum([t == p for t, p in zip(true, pred)])
        similarity.append(match_count)
    return similarity
def save_classification_report(y_true, y_pred, report_path):
    # Generate classification report
    report = classification_report(y_true, y_pred, target_names=configs.LABEL_SETS,digits=3)
    print(report)
    # Save to markdown file
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Classification Report\n")
        f.write("This is the classification report for the model's predictions.\n\n")
        f.write(report)

if __name__ == "__main__":
    predict_data = pd.read_excel(configs.PREDICT_DATA_PATH, sheet_name=configs.SHEET_NAME)[:]
    print(len(predict_data))
    print(configs.PREDICT_DATA_PATH)
    all_text = predict_data["原创微博"].tolist()
    batch_size = 32
    batch_num = len(all_text) // batch_size + 1
    results = []
    preds = []
    y_true = predict_data[configs.LABEL_SETS].values

    for i in tqdm(range(batch_num), desc="正在预测", total=batch_num):
        batch_text = all_text[i * batch_size : (i + 1) * batch_size]
        res, pred= predict(batch_text)
        results.extend(res)
        preds.extend(pred)
    # Compute similarity
    similarity = compute_similarity(y_true, preds)

    # Map similarity to new column values based on the described logic
    similarity_scores = []
    for sim in similarity:
            if sim == 5:
                similarity_scores.append(5)
            elif sim == 4:
                similarity_scores.append(4)
            elif sim == 3:
                similarity_scores.append(3)
            elif sim == 2:
                similarity_scores.append(2)
            else:
                similarity_scores.append(0)_
    # Add new column with similarity scores
    predict_data["一致性评分"] = similarity_scores
    predict_data["预测标签"] = results
    predict_data.to_excel(f"{configs.OUTPUTS_DIR}/{configs.BERT_DIR}_行为数据预测数据结果.xlsx", index=False)
    # Save the classification report
    save_classification_report(y_true, preds, f"{configs.OUTPUTS_DIR}/{configs.BERT_DIR}_classification_report.md")
