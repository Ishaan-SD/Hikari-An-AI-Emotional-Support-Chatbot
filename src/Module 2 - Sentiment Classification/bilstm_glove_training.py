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

def load_glove_embeddings(glove_file, tokenizer, embedding_dim):
    """
    Loads GloVe embeddings and creates an embedding matrix for our vocabulary.
    """
    embeddings_index = {}
    try:
        with open(glove_file, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32') # skipped first because its the word itself
                embeddings_index[word] = coefs
    except FileNotFoundError:
        print(f"Error: The GloVe file '{glove_file}' was not found.")
        return None
    
    print(f"Found {len(embeddings_index)} word vectors in GloVe file.")

    # Prepare embedding matrix
    word_index = tokenizer.word_index
    num_words = min(len(word_index) + 1, tokenizer.num_words)
    embedding_matrix = np.zeros((num_words, embedding_dim))
    
    for word, i in word_index.items():
        if i >= tokenizer.num_words:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # Words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
            
    return embedding_matrix

def train_bilstm_with_glove():

    input_filename = r'data\Cleaned Data\cleaned_dataset.csv' 
    glove_filename = r'data\glove.6B.100d.txt'
    embedding_dim = 100 # matching the dimension of the GloVe file (100d)
    
    # --- 1. Load and Prepare Data ---
    df = pd.read_csv(input_filename)
    df.dropna(subset=['Cleaned_Text'], inplace=True)
    
    X = df['Cleaned_Text']
    y = df['emotion']
    num_classes = len(y.unique())

    # --- 2. Tokenization and Sequencing ---
    vocab_size = 10000
    max_length = 200
    
    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(X)
    X_sequences = tokenizer.texts_to_sequences(X)
    X_padded = pad_sequences(X_sequences, maxlen=max_length, padding='post', truncating='post')

    # --- 3. Load GloVe Embeddings and Create Matrix ---
    embedding_matrix = load_glove_embeddings(glove_filename, tokenizer, embedding_dim)
    if embedding_matrix is None:
        return # Stop if GloVe file is not found

    # --- 4. Encode Labels and Split Data ---
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X_padded, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # --- 5. Build the BiLSTM Model with GloVe Embeddings ---
    
    model = Sequential()
    # The Embedding layer is now initialized with the GloVe weights
    model.add(Embedding(input_dim=embedding_matrix.shape[0],
                        output_dim=embedding_dim,
                        weights=[embedding_matrix],
                        input_length=max_length,
                        trainable=False)) # Freeze the embedding layer
    
    model.add(Bidirectional(LSTM(128, return_sequences=False)))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    # --- 6. Train the Model ---
    model.fit(X_train, y_train,
              epochs=10, # We can train for a bit longer now
              batch_size=64,
              validation_data=(X_test, y_test),
              verbose=1)
    
    # --- 7. Evaluate and Save ---
    print("\n--- Model Evaluation ---")
    y_pred_prob = model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy on the test set: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=encoder.classes_))
    
    output_dir = r'models\BiLSTM with Glove\bilstm_glove_model_merged'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    model.save(os.path.join(output_dir, 'bilstm_glove_model.keras'))
    joblib.dump(encoder, os.path.join(output_dir, 'label_encoder.joblib'))
    tokenizer_json = tokenizer.to_json()
    with open(os.path.join(output_dir, 'tokenizer.json'), 'w', encoding='utf-8') as f:
        f.write(json.dumps(tokenizer_json, ensure_ascii=False))
    

if __name__ == '__main__':
    train_bilstm_with_glove()
