import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json
import joblib
import os
import numpy as np

def train_bilstm_model():
    
    input_filename = r'data\Cleaned Data\cleaned_dataset.csv' 
    
    # --- 1. Load and Prepare Data ---
    try:
        df = pd.read_csv(input_filename)
    except FileNotFoundError:
        print(f"Error: The file '{input_filename}' was not found.")
        return
        
    df.dropna(subset=['Cleaned_Text'], inplace=True)
    
    X = df['Cleaned_Text']
    y = df['emotion']
    num_classes = len(y.unique())
    print(f"Found {num_classes} unique emotion classes.")

    # --- 2. Tokenization and Sequencing ---
    vocab_size = 10000
    max_length = 200
    
    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(X)
    
    X_sequences = tokenizer.texts_to_sequences(X)
    
    # --- 3. Padding Sequences ---
    X_padded = pad_sequences(X_sequences, maxlen=max_length, padding='post', truncating='post')

    # --- 4. Encode Labels ---
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    
    # --- 5. Train-Test Split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X_padded, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # --- 6. Build the Bidirectional LSTM Model ---
    
    embedding_dim = 128
    
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length))
    
    # CHANGE: The LSTM layer is now wrapped in a Bidirectional layer.
    model.add(Bidirectional(LSTM(128, return_sequences=False)))
    
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
                  
    model.summary()

    # --- 7. Train the Model ---
    history = model.fit(X_train, y_train,
                        epochs=5,
                        batch_size=64,
                        validation_data=(X_test, y_test),
                        verbose=1)
    
    # --- 8. Evaluate Performance ---
    print("\n--- Model Evaluation ---")
    y_pred_prob = model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy on the test set: {accuracy:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=encoder.classes_))
    
    # --- 9. Save the Model and Helper Files ---
    output_dir = r'models\BiLSTM\bilstm_model_merged' # Save to a new directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    model_path = os.path.join(output_dir, 'bilstm_model.keras')
    tokenizer_path = os.path.join(output_dir, 'tokenizer.json')
    encoder_path = os.path.join(output_dir, 'label_encoder.joblib')
    
    model.save(model_path)
    
    tokenizer_json = tokenizer.to_json()
    with open(tokenizer_path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(tokenizer_json, ensure_ascii=False))

    joblib.dump(encoder, encoder_path)


if __name__ == '__main__':
    train_bilstm_model()
