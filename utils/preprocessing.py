import re
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Label Mapping
label_map = {
    "sadness": 0,
    "joy": 1,
    "love": 2,
    "anger": 3,
    "fear": 4,
    "surprise": 5
}

# -------------------------
# CLEAN TEXT
# -------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text

# -------------------------
# LOAD DATA
# -------------------------
def load_data(file_path):
    texts, labels = [], []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            text, label = line.strip().split(";")
            texts.append(clean_text(text))
            labels.append(label_map[label])

    return texts, np.array(labels)

# -------------------------
# TOKENIZER
# -------------------------
def create_tokenizer(texts, max_vocab=10000):
    tokenizer = Tokenizer(num_words=max_vocab, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    return tokenizer

# -------------------------
# TEXT TO PADDED SEQUENCES
# -------------------------
def texts_to_padded(tokenizer, texts, max_len=30):
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=max_len, padding='post')
    return padded
