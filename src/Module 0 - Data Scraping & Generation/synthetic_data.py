import google.generativeai as genai
import os
import pandas as pd

genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

try:
    gemini_model_for_synthetic_data = genai.GenerativeModel('gemini-2.5-pro')
except Exception as e:
    print(f"Failed to initialize Gemini model: {e}")
    print('\nExiting...')
    exit()

def generate_synthetic_data(num_samples, emotion_type, age_group="general"):

    prompt = f"""
    Generate {num_samples} short, realistic text snippets expressing the emotion '{emotion_type}'.
    The snippets should be typical of someone from the '{age_group}' age group.
    Each snippet should be on a new line. Do not add any extra formatting or numbering.
    Focus purely on expressing the emotion naturally.
    Example for sadness (adult):
    I just can't seem to shake this feeling of gloom today.
    Everything feels so heavy right now.
    I wish things were different.

    Example for joy (5-year-old):
    My mommy gave me a big hug!
    I love my new toy car!
    Playing with my friends makes me so happy!

    Now, generate {num_samples} snippets for '{emotion_type}' ({age_group}):
    """
    
    generated_texts = []
    attempts = 0
    max_attempts = 3 # Try again if the generation fails or returns too few samples

    while len(generated_texts) < num_samples and attempts < max_attempts:
        attempts += 1
        print(f"  Attempt {attempts} to generate {num_samples - len(generated_texts)} samples for '{emotion_type}'...")
        try:
            response = gemini_model_for_synthetic_data.generate_content(
                prompt,
                generation_config={"max_output_tokens": 1000} # length of the responses
            )
            if hasattr(response, 'text') and response.text:
                new_texts = [line.strip() for line in response.text.split('\n') if line.strip()]
                generated_texts.extend(new_texts)
                # Remove duplicates if present
                generated_texts = list(set(generated_texts))
                
                if len(generated_texts) >= num_samples:
                    print(f"  Successfully generated {len(generated_texts)} samples for '{emotion_type}'.")
                    return generated_texts[:num_samples] # quota fulfilled and return requied samples
            else:
                print(f"  Gemini returned an empty or invalid response for {emotion_type} in attempt {attempts}.")

        except Exception as e:
            print(f"  Error generating synthetic data for {emotion_type} in attempt {attempts}: {e}")
        
        prompt += f"\n\nPlease generate {num_samples - len(generated_texts)} more unique snippets for '{emotion_type}'."


    print(f"  Warning: Could only generate {len(generated_texts)} out of {num_samples} for '{emotion_type}' after {max_attempts} attempts.")
    return generated_texts[:num_samples] 


def create_balanced_emotion_dataset_synthetic(samples_per_emotion: int = 100) -> pd.DataFrame:
    """
    Creates a balanced dataset for emotion classification purely using
    synthetic data generation via Gemini.
    """

    if gemini_model_for_synthetic_data is None:
        print("Cannot create balanced dataset: Gemini model not initialized.")
        return pd.DataFrame()

    print(f"\nCreating balanced synthetic dataset with {samples_per_emotion} samples per emotion...")

    # Define all target emotions
    all_target_emotions = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise', 'disgust', 'neutral']
    
    balanced_data = []
    
    # Process each emotion
    for emotion_name in all_target_emotions:
        print(f"\nGenerating {samples_per_emotion} samples for emotion: '{emotion_name}'...")
        synthetic_texts = generate_synthetic_data(samples_per_emotion, emotion_name, age_group="general")
        
        for text in synthetic_texts:
            balanced_data.append({'text': text, 'emotion': emotion_name})
        
        print(f"  Added {len(synthetic_texts)} samples for '{emotion_name}'.")

    # Create DataFrame and Shuffle
    df_balanced = pd.DataFrame(balanced_data)

    # Shuffle the entire dataset
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"\n--- Balanced Synthetic Dataset Created ---")
    print(f"Total samples: {len(df_balanced)}")
    print("Distribution:")
    print(df_balanced['emotion'].value_counts())

    return df_balanced

if __name__ == "__main__":

    target_samples = 3000

    balanced_df = create_balanced_emotion_dataset_synthetic(samples_per_emotion=target_samples)

    if not balanced_df.empty:
        print("\nFirst 5 rows of the balanced DataFrame:")
        print(balanced_df.head())

        file_name = f"balanced_synthetic_emotion_dataset.csv"
        balanced_df.to_csv(file_name, index=False)
        print(f"\nBalanced synthetic dataset saved to '{file_name}'")
    else:
        print("\nFailed to create balanced dataset.")