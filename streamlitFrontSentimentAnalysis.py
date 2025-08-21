import streamlit as st
import requests
from JackageNormalizer import normalize_persian_text

st.set_page_config(
    layout="centered", page_title="SentimentAnalysis", page_icon="🧠"
)

API_KEY = "Fill in this section with your own key."
URL = "https://api.sapling.ai/api/v1/spellcheck"

# API Function to get spellcheck Call
def get_spellcheck_suggestions(text, URL=URL, API_KEY=API_KEY):
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

def correction_ui():
    
    normalized_text = st.session_state.get("normalized_text", "")
    suggestions = st.session_state.get("suggestions", [])
    
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
        
def user_input_features():
    user_text = st.text_area("write your text: ")

    if st.button("analyze", key="analyze_button"):
        if user_text.strip() == "":
            st.warning("Please enter some text.")
        else:
            normalized_text = normalize_persian_text(user_text)
            st.markdown(f"Normalized output is : {normalized_text}")
            
            st.session_state["normalized_text"] = normalized_text
            st.session_state["suggestions"] = get_spellcheck_suggestions(normalized_text)
            st.session_state.final_text = ""

# Streamlit UI
st.title("NLP : SentimentAnalysis")

st.write("""
### SentimentAnalysis with Transformers

The model used in this project is a Persian transformer model called ParsBERT,\
    which has been fine-tuned for the purpose of sentiment analysis based on Persian data.
    """)

text = user_input_features()

if "suggestions" in st.session_state and st.session_state["suggestions"]:
    correction_ui()

article_text = """
@article{ParsBERT,
    title={ParsBERT: Transformer-based Model for Persian Language Understanding},
    author={Mehrdad Farahani, Mohammad Gharachorloo, Marzieh Farahani, Mohammad Manthouri},
    journal={ArXiv},
    year={2020},
    volume={abs/2005.12515}
}
"""

st.code(article_text, language="bibtex") 

#test output from the API call:
#[{'end': 17,
#  'id': '59e5fc1f-940c-5e98-a91b-11984083920b',
#  'replacement': 'spelling',
#  'sentence': 'There are speling mistakes here.',
#  'sentence_start': 0,
#  'start': 10}]
