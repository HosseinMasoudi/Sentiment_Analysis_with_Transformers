import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import tensorflow as tf
import numpy as np
import streamlit as st
import requests

from JackageNormalizer import normalize_persian_text
from tensorflow import keras
from transformers import TFBertModel
from transformers import AutoTokenizer


st.set_page_config(
    layout="centered", page_title="SentimentAnalysis", page_icon="🧠"
)

API_KEY = "R80X16GG098GCNV5FR99PP9UC4BNKWQI"
URL = "https://api.sapling.ai/api/v1/spellcheck"
MODEL_NAME = "HooshvareLab/bert-fa-base-uncased"
MODEL_PATH = "/Users/hossein/Desktop/Hos/CODE/NLP/SentimentAnalyis_model/saved_model_tf"
threshold = 0.5

# Load tokenizer and model
@st.cache_resource(show_spinner=True)
def load_model_and_tokenizer(model_path, model_name):
    
    model = keras.models.load_model(model_path, custom_objects={"TFBertModel": TFBertModel})
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    return model, tokenizer

try:
    loaded_model, model_tokenizer = load_model_and_tokenizer(MODEL_PATH, MODEL_NAME)
    st.sidebar.success("Model and Tokenizer loaded")
except Exception as e:
    st.sidebar.error(f"Loading failed:\n {e}")
    st.stop()

def encode_texts(texts, tokenizer=model_tokenizer, max_len=64):

    encode = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="np",
    )
    encode["input_ids"] = encode["input_ids"].astype("int32")
    encode["attention_mask"] = encode["attention_mask"].astype("int32")
    
    return encode

def predict_text(text : str):
    normalize_text = normalize_persian_text(text)
    encode = encode_texts(normalize_text)

    probability = float(
        loaded_model.predict(
            {"input_ids": encode["input_ids"], "attention_mask": encode["attention_mask"]},
            verbose=0,
        ).reshape(-1)[0]
    )

    label = "POSITIVE" if probability >= threshold else "NEGATIVE"
    return probability, label


# API Function to get spellcheck Call
def get_spellcheck_suggestions(text : str, URL=URL, API_KEY=API_KEY):
    payload = {
        "key": API_KEY,
        "text": text,
        "session_id": "persian-spellcheck",
        "lang": "fa"
    }
    
    try:
        response = requests.post(URL, json=payload)

        if 200 <= response.status_code < 300:
            resp_json = response.json()
            return resp_json.get("edits", [])
        else:
            st.error(f"API Error {response.status_code}: {response.text}")
            return []

    except Exception as e:
        st.error(f"Error calling Sapling API: {e}")
        return []
    
def get_current_text(user_text: str):
    """If final_text exists and not empty, use it; otherwise use user_text"""
    final_txt = st.session_state.get("final_text", "")
    return final_txt if isinstance(final_txt, str) and final_txt.strip() else user_text
    
# Streamlit UI
st.title("NLP : SentimentAnalysis")

st.write("""
### SentimentAnalysis with Transformers

The model used in this project is a Persian transformer model called ParsBERT,\
    which has been fine-tuned for the purpose of sentiment analysis based on Persian data.
    """)

     
def user_input_features():
    user_text = st.text_area("write your text in persian: ")

    if st.button("analyze", key="analyze_button"):
        if user_text.strip() == "":
            st.warning("Please enter some text.")
            return

        normalized_text = normalize_persian_text(user_text)
        st.markdown(f"Normalized output is : {normalized_text}")

        st.session_state["normalized_text"] = normalized_text
        st.session_state["suggestions"] = get_spellcheck_suggestions(normalized_text)
        st.session_state.final_text = ""

        text_for_pred = get_current_text(user_text)
        probability, label = predict_text(text_for_pred)
        
        if label == "NEGATIVE":
            st.markdown(
                f"**Prediction on text :** <span style='color:red'>{label}</span> (score: `{probability:.4f}`)",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"**Prediction on text :** <span style='color:green'>{label}</span> (score: `{probability:.4f}`)",
                unsafe_allow_html=True
            )

def correction_ui():
    
    normalized_text = st.session_state.get("normalized_text", "")
    if not normalized_text :
        return
    suggestions = st.session_state.get("suggestions", [])
    if not suggestions :
        st.info("No spellcheck suggestions.")
        return
    
    suggestions = sorted(suggestions, key=lambda s: s["start"], reverse=True)
    
    for suggestion in suggestions:
        start = suggestion["start"]
        end = suggestion["end"]
        orginal_word = normalized_text[start:end]
        replacement = suggestion['replacement']
              
        choice_key = f"choice_{suggestion['id']}"  
        choice = st.radio(
            f"Should '{replacement}' replace '{orginal_word}'?",
            options=["Yes", "No"],
            key=choice_key,
            index=None
        )
        
    choices = {k: v for k, v in st.session_state.items() if k.startswith("choice_")}   
    if len(choices) == len(suggestions) and all(v in ["Yes","No"] for v in choices.values()):
        corrected_list = list(normalized_text)
        for suggestion in suggestions:
            start = suggestion["start"]
            end = suggestion["end"]
            replacement = suggestion['replacement']
            if st.session_state[f"choice_{suggestion['id']}"] == "Yes":
                corrected_list[start:end] = list(replacement)
        st.session_state.final_text = "".join(corrected_list)
        
        st.markdown(f"**Final corrected text:**\n {st.session_state.final_text}")
        if st.session_state["final_text"].strip():
            text_for_pred = get_current_text("")
            probability2, label2 = predict_text(text_for_pred) 
            if label2 == "NEGATIVE":
                st.markdown(
                    f"**Prediction on text :** <span style='color:red'>{label2}</span> (score: `{probability2:.4f}`)",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"**Prediction on text :** <span style='color:green'>{label2}</span> (score: `{probability2:.4f}`)",
                    unsafe_allow_html=True
                )

user_input_features()
if "suggestions" in st.session_state and st.session_state["suggestions"]:
    correction_ui()

#test output from the API call:
#[{'end': 17,
#  'id': '59e5fc1f-940c-5e98-a91b-11984083920b',
#  'replacement': 'spelling',
#  'sentence': 'There are speling mistakes here.',
#  'sentence_start': 0,
#  'start': 10}]
