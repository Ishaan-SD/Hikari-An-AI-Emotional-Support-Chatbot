import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

def train_ml_model():

    input_filename = r'data\Cleaned Data\cleaned_dataset.csv'
    
    # --- 1. Load Cleaned Data ---
    df = pd.read_csv(input_filename)
        
    # Drop rows where 'Cleaned_Text' is empty, as this can cause issues.
    df.dropna(subset=['Cleaned_Text'], inplace=True)
    
    # Define our features (X) and target (y)
    X = df['Cleaned_Text']
    y = df['emotion']

    # --- 2. Feature Extraction (TF-IDF) ---
    print("Converting text data to numerical vectors using TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=5000)
    
    # --- 3. Train-Test Split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Fit the vectorizer on the training data and transform it.
    X_train_tfidf = vectorizer.fit_transform(X_train)
    # Only transform the test data using the already-fitted vectorizer.
    X_test_tfidf = vectorizer.transform(X_test)
    
    # --- 4. Define and Train Models ---
    models = {
        "Logistic Regression": LogisticRegression(class_weight='balanced', max_iter=1000),
        "Multinomial Naive Bayes": MultinomialNB(),
        "Linear SVC": LinearSVC(class_weight='balanced', max_iter=2000),
        "Random Forest": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    }
    
    best_model = None
    best_accuracy = 0.0
    
    for name, model in models.items():
        print(f"\n--- Training {name} ---")
        model.fit(X_train_tfidf, y_train)
        print(f"Training complete.")
        
        print(f"\n--- Evaluating {name} ---")
        predictions = model.predict(X_test_tfidf)
        accuracy = accuracy_score(y_test, predictions)
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, predictions))
        
        # Check if this is the best model so far
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            print(f"*** New best model found: {name} ***")

    # --- 5. Save the Best Model and Vectorizer ---
    if best_model is not None:
        output_dir = r'models\ML\ml_model_merged'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        model_path = os.path.join(output_dir, 'ml_model.joblib')
        vectorizer_path = os.path.join(output_dir, 'tfidf_vectorizer.joblib')
        
        print(f"\n--- Saving Best Model: {best_model.__class__.__name__} ---")
        print(f"Best accuracy was: {best_accuracy:.4f}")
        
        print(f"Saving model to '{model_path}'...")
        joblib.dump(best_model, model_path)
        
        print(f"Saving vectorizer to '{vectorizer_path}'...")
        joblib.dump(vectorizer, vectorizer_path)
        
    else:
        print("\nCould not determine the best model. No model was saved.")

if __name__ == '__main__':
    train_ml_model()
