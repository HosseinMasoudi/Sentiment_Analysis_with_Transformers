import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import tensorflow as tf
from tensorflow import keras
from transformers import TFBertModel
from transformers import AutoTokenizer

model_path = "saved_model_tf"

loaded_model = keras.models.load_model(
    model_path,
    custom_objects={"TFBertModel": TFBertModel}
)

print("Model loaded successfully")
loaded_model.summary()


tokenizer = AutoTokenizer.from_pretrained("HooshvareLab/bert-fa-base-uncased")

texts = [
    "این غذا خیلی عالی بود و واقعا دوستش داشتم!",
    "کاملاً ناامید شدم، اصلاً خوب نبود."
]

encoder = tokenizer(
    texts,
    padding=True,
    truncation=True,
    max_length=64,
    return_tensors="np"
)

preds = loaded_model.predict(
    {"input_ids": encoder["input_ids"].astype("int32"),
     "attention_mask": encoder["attention_mask"].astype("int32")},
    verbose=0
)

for t, p in zip(texts, preds.reshape(-1)):
    print(f"Text: {t}")
    print(f"Prediction score (≈1 posetive / ≈0 negative): {float(p):.4f}\n")