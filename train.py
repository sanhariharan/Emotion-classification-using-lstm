import tensorflow as tf
from model.data_loader import prepare_data
from model.emotion_classifier import build_model
from utils.metrics import evaluate_model

# -------------------------
# CONFIG
# -------------------------
TRAIN_PATH = "data/train.txt"
VAL_PATH = "data/val.txt"
TEST_PATH = "data/test.txt"

MAX_VOCAB = 10000
MAX_LEN = 30
EMBED_DIM = 128
BATCH_SIZE = 32
EPOCHS = 10

# -------------------------
# LOAD DATA
# -------------------------
(X_train, y_train,
 X_val, y_val,
 X_test, y_test,
 tokenizer, vocab_size) = prepare_data(
    TRAIN_PATH, VAL_PATH, TEST_PATH, MAX_VOCAB, MAX_LEN
)

print("Data Loaded!")
print("Vocab Size:", vocab_size)

# -------------------------
# BUILD MODEL
# -------------------------
model = build_model(vocab_size, MAX_LEN, EMBED_DIM)

model.summary()

# -------------------------
# CALLBACKS
# -------------------------
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=2,
    restore_best_weights=True
)


# TRAIN

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop]
)

# -------------------------
# SAVE MODEL
# -------------------------
model.save("emotion_model.h5")
print("Model Saved!")

# -------------------------
# EVALUATE
# -------------------------
loss, acc = model.evaluate(X_test, y_test)
print(f"\nTest Accuracy: {acc:.4f}")

evaluate_model(model, X_test, y_test)
