import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report, accuracy_score

# Load the test dataset
test_file_path = 'D:/hehai/textCNN/政府测试.xlsx'  # Update this path
test_df = pd.read_excel(test_file_path)

# Extract text data
X_test = test_df['原创微博'].values

# Load the trained model
model = load_model('textcnn_model_best_government.keras')

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
y_true = test_df[['Command and coordination', 'Personnel rescue and relocation', 'Facility defense and repairs',
                  'Information release and crisis communication', 'Recovery of economy and life']].values

# Generate classification report with zero_division=0 to avoid undefined metrics
classification_rep = classification_report(y_true, predicted_labels, target_names=[
    'Command and coordination', 'Personnel rescue and relocation', 'Facility defense and repairs',
    'Information release and crisis communication', 'Recovery of economy and life'], output_dict=True, zero_division=0)

# Prepare the classification metrics in the desired format
metrics = classification_rep.copy()

# Extract metrics and round them to 3 decimal places
formatted_metrics = {
    "precision": [round(metrics[label]['precision'], 3) if label != 'accuracy' else None for label in metrics],
    "recall": [round(metrics[label]['recall'], 3) if label != 'accuracy' else None for label in metrics],
    "f1-score": [round(metrics[label]['f1-score'], 3) if label != 'accuracy' else None for label in metrics],
    "support": [round(metrics[label]['support'], 3) if label != 'accuracy' else None for label in metrics]
}

# Calculate accuracy from model predictions
accuracy = accuracy_score(y_true, predicted_labels)

# Add accuracy to the formatted metrics
formatted_df = pd.DataFrame(formatted_metrics, index=[label for label in metrics])
formatted_df.loc['accuracy'] = [round(accuracy, 3)] * 4  # Add accuracy row, rounded to 3 decimals

# Display the formatted metrics without truncation
pd.set_option('display.max_rows', None)  # Ensure no rows are truncated
pd.set_option('display.max_columns', None)  # Ensure no columns are truncated
pd.set_option('display.width', None)  # Prevent line breaks
pd.set_option('display.max_colwidth', None)  # Prevent column content truncation

# Display the formatted metrics
print(formatted_df)
