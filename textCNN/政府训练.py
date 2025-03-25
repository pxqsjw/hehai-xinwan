import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D, Embedding, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load the dataset
train_file_path = 'D:/hehai/textCNN/政府训练.xlsx'  # Update this path
train_df = pd.read_excel(train_file_path)

# Extract text and labels
X = train_df['原创微博'].values
y = train_df[['Command and coordination', 'Personnel rescue and relocation', 'Facility defense and repairs',
        'Information release and crisis communication', 'Recovery of economy and life']].values

# Tokenize the text data
tokenizer = Tokenizer(num_words=10000)  # Top 10,000 words
tokenizer.fit_on_texts(X)
X_seq = tokenizer.texts_to_sequences(X)

# Pad the sequences to make them the same length
X_pad = pad_sequences(X_seq, maxlen=100)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_pad, y, test_size=0.2, random_state=42)

# Build the TextCNN model
model = Sequential()
model.add(Embedding(input_dim=10000, output_dim=128, input_length=100))
model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dense(5, activation='sigmoid'))  # 5 output labels

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Save the model during training with the best validation accuracy
checkpoint = ModelCheckpoint('textcnn_model_best_government.keras', monitor='val_accuracy', save_best_only=True, mode='max')

# Train the model
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test), callbacks=[checkpoint])

# Save the final model
model.save('textcnn_model_final_government.keras')

