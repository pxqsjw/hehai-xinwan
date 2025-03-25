import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
import pandas as pd
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout

from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint


# Load the dataset
df = pd.read_excel('D:/hehai/LSTM/政府训练.xlsx')

# Extract the input (text) and output columns
X = df['原创微博'].values
y = df[['Command and coordination', 'Personnel rescue and relocation', 'Facility defense and repairs',
        'Information release and crisis communication', 'Recovery of economy and life']].values

# Tokenize the text data
tokenizer = Tokenizer(num_words=10000)  # Consider top 10,000 words
tokenizer.fit_on_texts(X)
X_seq = tokenizer.texts_to_sequences(X)

# Pad the sequences to make them the same length
X_pad = pad_sequences(X_seq, maxlen=100)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_pad, y, test_size=0.2, random_state=42)

# Build the LSTM model
model = Sequential()
model.add(Embedding(input_dim=10000, output_dim=128, input_length=100))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(64, activation='relu'))
model.add(Dense(5, activation='sigmoid'))  # 5 output labels

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Update the file format to '.keras'
checkpoint = ModelCheckpoint('lstm_model_best.keras', monitor='val_accuracy', save_best_only=True, mode='max')

# Train the model
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test), callbacks=[checkpoint])

# Save the final model
model.save('lstm_model_final_government.h5')

# Optionally, you can also save the tokenizer if you want to use it later
import pickle
with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)
