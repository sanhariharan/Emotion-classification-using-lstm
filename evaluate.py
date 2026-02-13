import tensorflow as tf
from model.data_loader import prepare_data
from utils.metrics import evaluate_model

# Paths
TEST_PATH = "data/test.txt"
TRAIN_PATH = "data/train.txt"
VAL_PATH = "data/val.txt"

# Load Data
(_, _,
 _, _,
 X_test, y_test,
 tokenizer, vocab_size) = prepare_data(
    TRAIN_PATH, VAL_PATH, TEST_PATH
)

# Load Model
model = tf.keras.models.load_model("emotion_model.h5")

# Evaluate
evaluate_model(model, X_test, y_test)
