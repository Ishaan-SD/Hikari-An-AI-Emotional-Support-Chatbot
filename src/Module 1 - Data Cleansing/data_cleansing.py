# import nltk
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')

import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk

def clean_text(text):
    """
    A function that will clean the passed text.
    """
    # We will be doing basic cleaning to avoid too much loss of context

    # 1. lowercase
    text = text.lower()
    
    # 2. Remove punctuation and special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # 3. Tokenizing
    tokens = word_tokenize(text)
    
    # 4. Removing stop words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # 5. Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    # Joining the tokens and then sending back as a string.
    return ' '.join(tokens)

def main():

    input_filename = r'data\Raw data\merged_data.csv'
    output_filename = r'data\Cleaned Data\cleaned_merged_dataset.csv'

    print(f"Loading data from '{input_filename}'...")
    try:
        df = pd.read_csv(input_filename)
    except FileNotFoundError:
        print(f"Error: The file '{input_filename}' was not found.")
        return
        
    # Drop rows where the 'Text' column is empty or not a string
    df.dropna(subset=['Text'], inplace=True)
    df = df[df['Text'].apply(lambda x: isinstance(x, str))]

    print("Cleaning the 'Text' column... This may take a moment.")
    
    # Apply the cleaning function to each entry in the 'Text' column
    df['Cleaned_Text'] = df['Text'].apply(clean_text)

    df.to_csv(output_filename, index=False)

if __name__ == '__main__':

    main()
