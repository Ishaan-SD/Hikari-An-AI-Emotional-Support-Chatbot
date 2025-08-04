
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import joblib
import os
import numpy as np

def train_ann_model():

    # Use the original cleaned file with multiple emotions
    input_filename = r'data\Cleaned Data\cleaned_dataset.csv' 
    
    # --- 1. Load and Prepare Data ---
    print(f"Loading cleaned data from '{input_filename}'...")
    try:
        df = pd.read_csv(input_filename)
    except FileNotFoundError:
        print(f"Error: The file '{input_filename}' was not found.")
        print("Please make sure you have run the 'data_cleaning.py' script first.")
        return
        
    df.dropna(subset=['Cleaned_Text'], inplace=True)
    
    X = df['Cleaned_Text']
    y = df['emotion']

    num_classes = len(y.unique())
    print(f"Found {num_classes} unique emotion classes.")

    # --- 2. Feature Extraction (TF-IDF) ---
    print("Converting text data to numerical vectors using TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=5000)
    X_tfidf = vectorizer.fit_transform(X).toarray()

    # --- 3. Encode Labels ---
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    
    # --- 4. Train-Test Split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X_tfidf, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # --- 5. Build the ANN Model for Multi-Class Classification ---
    
    input_dim = X_train.shape[1] 
    
    model = Sequential()
    model.add(Dense(128, input_dim=input_dim, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
                  
    model.summary()

    # --- 6. Train the Model ---
    print("\nTraining the ANN model...")
    history = model.fit(X_train, y_train,
                        epochs=10, # Increased epochs for better learning on a more complex task
                        batch_size=32,
                        validation_data=(X_test, y_test),
                        verbose=1)
    print("Model training complete.")
    
    # --- 7. Evaluate Performance ---
    print("\n--- Model Evaluation ---")
    # For multi-class, we need to get the class with the highest probability
    y_pred_prob = model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy on the test set: {accuracy:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=encoder.classes_))
    
    # --- 8. Save the Model, Vectorizer, and Encoder ---
    output_dir = r'D:\CDAC\Final Project\Hikari-An-AI-Emotional-Support-Chatbot\models\ANN\ann_model_multiclass_merged'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    model_path = os.path.join(output_dir, 'ann_model.keras')
    vectorizer_path = os.path.join(output_dir, 'tfidf_vectorizer.joblib')
    encoder_path = os.path.join(output_dir, 'label_encoder.joblib')
    
    print(f"\nSaving model to '{model_path}'...")
    model.save(model_path)
    
    print(f"Saving vectorizer to '{vectorizer_path}'...")
    joblib.dump(vectorizer, vectorizer_path)
    
    print(f"Saving label encoder to '{encoder_path}'...")
    joblib.dump(encoder, encoder_path)
    
    print("\n--- Multi-Class ANN Training Process Finished Successfully ---")

if __name__ == '__main__':
    train_ann_model()
