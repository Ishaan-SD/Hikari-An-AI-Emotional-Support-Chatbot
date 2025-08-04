import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
import requests
import os
import json
import pandas as pd

# --- 1. Configuration and Model Loading ---

st.set_page_config(
    page_title="Hikari - Your AI Companion",
    page_icon="💖"
)

@st.cache_resource
def load_classifier_model():


    model_dir = r"models\Transformer\transformer_model_merged"
    # model_dir = "./transformer_model"
    if not os.path.exists(model_dir):
        st.error(f"Model directory not found at {model_dir}. Please run transformer_training.py first.")
        return None, None
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        model.eval()
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading the classification model: {e}")
        return None, None

tokenizer, classifier_model = load_classifier_model()

# --- 2. Emotion Prediction Function ---

def predict_emotion(text):
    """
    Predicts the emotion of a given text using the loaded classifier.
    """
    if not text or not tokenizer or not classifier_model:
        return "neutral"

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    
    with torch.no_grad():
        logits = classifier_model(**inputs).logits
        
    predicted_class_id = torch.argmax(logits, dim=1).item()
    emotion = classifier_model.config.id2label[predicted_class_id]
    return emotion

# --- 3. Gemini API Call Function ---

def get_gemini_response(user_input, emotion, character_name):
    """
    Calls the Gemini API with an engineered prompt based on the detected emotion
    and an optional character persona.
    """
    # Start with the base prompt
    base_prompt = (
        f"You are Hikari, a compassionate and empathetic AI emotional support companion. "
        f"A user is feeling '{emotion}'. Their message is: '{user_input}'. "
        f"Act like a therapist."
        f"Try to respond within 4-5 lines."
        f"Try to respond normally for neutral emotion."
        f"You can provide some helping technique only if available."
    )
    
    # NEW: Add character persona instructions if a character is provided
    if character_name:
        persona_prompt = (
            f"Please respond in the tone and style of the character '{character_name}'. "
            f"However, your core message MUST remain gentle, validating, and supportive like a therapist. "
            f"Do not give medical advice. Focus on listening and showing you understand, but through that character's voice."
        )
    else:
        persona_prompt = (
            "Please provide a gentle, validating, and supportive response. "
            "Do not give medical advice. Focus on listening and showing you understand."
        )
        
    full_prompt = base_prompt + persona_prompt
    
    api_key = os.getenv("GOOGLE_API_KEY")
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
    
    payload = {"contents": [{"parts": [{"text": full_prompt}]}]}
    
    try:
        response = requests.post(api_url, json=payload, headers={'Content-Type': 'application/json'})
        response.raise_for_status()
        result = response.json()
        
        if (result.get("candidates") and result["candidates"][0].get("content") and 
            result["candidates"][0]["content"].get("parts")):
            return result["candidates"][0]["content"]["parts"][0]["text"]
        else:
            return "I'm here for you. Could you tell me a little more?"

    except requests.exceptions.RequestException as e:
        print(f"Error calling Gemini API: {e}")
        return "I'm having a little trouble connecting right now, but please know that I'm still here to listen."


# --- 4. Streamlit UI and Chat Logic ---

st.title("💖 Hikari - Your AI Companion")
st.write("This is a safe space. I'm here to listen without judgment.")

# NEW: Add a sidebar for character selection
with st.sidebar:
    st.header("Customization")
    character_name_input = st.text_input(
        "Enter your favorite character (optional)",
        key="character_name",
        help="Hikari will try to respond in the tone of this character."
    )

st.write("---")

if classifier_model:
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "How are you feeling today?"}]
    if "emotion_history" not in st.session_state:
        st.session_state.emotion_history = []

    # Display past messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle user input
    if prompt := st.chat_input("Tell me what's on your mind..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Classify the emotion
                detected_emotion = predict_emotion(prompt)
                st.write(f"_Emotion detected: {detected_emotion}_")
                
                st.session_state.emotion_history.append(detected_emotion)
                
                # Get response from Gemini API, now with character name
                response = get_gemini_response(prompt, detected_emotion, st.session_state.character_name)
                st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()

else:
    st.info("The chatbot is currently unavailable. Please ensure the classification model is trained and available.")

# --- 5. Emotion Visualization Section ---
if len(st.session_state.get("emotion_history", [])) > 1:
    with st.expander("Track Your Emotion Journey", expanded=False):
        st.write("Here's a look at how your emotions have fluctuated during our conversation.")
        
        # scores for plotting emotions
        emotion_scores = {
            'happy': 2, 'surprised': 1, 'neutral': 0, 'fearful': -1, 
            'sad': -1, 'regret': -1, 'angry': -1, 'disgusted': -1, 
            'depressed': -2, 'love': 2
        }
        
        
        history_df = pd.DataFrame({
            'Emotion': st.session_state.emotion_history,
            'Sentiment Score': [emotion_scores.get(e, 0) for e in st.session_state.emotion_history]
        })
        history_df['Message Number'] = range(1, len(history_df) + 1)
        
        st.line_chart(history_df, x='Message Number', y='Sentiment Score')
        st.dataframe(history_df[['Message Number', 'Emotion']], use_container_width=True)
