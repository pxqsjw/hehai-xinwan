import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report, accuracy_score

# Load the test dataset
test_file_path = 'D:/hehai/textCNN/公众测试.xlsx'  # Update this path
test_df = pd.read_excel(test_file_path)

# Extract text data
X_test = test_df['原创微博'].values

# Load the trained model
model = load_model('textcnn_model_best.keras')

# Tokenize the text data
tokenizer = Tokenizer(num_words=10000)  # Use the same tokenizer
tokenizer.fit_on_texts(X_test)
X_seq = tokenizer.texts_to_sequences(X_test)

# Pad the sequences to make them the same length
X_pad = pad_sequences(X_seq, maxlen=100)

# Predict the labels
predictions = model.predict(X_pad)
predicted_labels = (predictions > 0.5).astype(int)  # Threshold at 0.5 for multi-label classification

# Display the classification report
y_true = test_df[['Problem solving', 'Instrumental social support offering',
                  'Emotional social support offering', 'Escape and vent',
                  'Egoism and Criminality']].values

# Generate classification report with zero_division=0 to avoid undefined metrics
classification_rep = classification_report(y_true, predicted_labels, target_names=[
    'Problem solving', 'Instrumental social support offering',
    'Emotional social support offering', 'Escape and vent',
    'Egoism and Criminality'], output_dict=True, zero_division=0)

# Prepare the classification metrics in the desired format
metrics = classification_rep.copy()

# Format metrics to keep 3 decimal places
formatted_metrics = {
    "precision": [round(metrics[label]['precision'], 3) for label in metrics if label != 'accuracy'],
    "recall": [round(metrics[label]['recall'], 3) for label in metrics if label != 'accuracy'],
    "f1-score": [round(metrics[label]['f1-score'], 3) for label in metrics if label != 'accuracy'],
    "support": [metrics[label]['support'] for label in metrics if label != 'accuracy']
}

# Calculate accuracy from model predictions
accuracy = accuracy_score(y_true, predicted_labels)

# Add accuracy to the formatted metrics
formatted_df = pd.DataFrame(formatted_metrics, index=[label for label in metrics if label != 'accuracy'])
formatted_df.loc['accuracy'] = [round(accuracy, 3)] * 4  # Add accuracy row with 3 decimal places

# Display the formatted metrics
print("Formatted Classification Report:")
print(formatted_df)

# Optionally, you can save the updated dataframe to a new Excel file
formatted_df.to_excel('textcnn_classification_report_test_public.xlsx', index=True)
