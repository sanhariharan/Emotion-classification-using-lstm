from utils.preprocessing import load_data, create_tokenizer, texts_to_padded

def prepare_data(train_path, val_path, test_path, max_vocab=10000, max_len=30):

    # Load data
    train_texts, train_labels = load_data(train_path)
    val_texts, val_labels = load_data(val_path)
    test_texts, test_labels = load_data(test_path)

    # Tokenizer
    tokenizer = create_tokenizer(train_texts, max_vocab)

    # Convert to padded sequences
    X_train = texts_to_padded(tokenizer, train_texts, max_len)
    X_val = texts_to_padded(tokenizer, val_texts, max_len)
    X_test = texts_to_padded(tokenizer, test_texts, max_len)

    vocab_size = min(max_vocab, len(tokenizer.word_index) + 1)

    return (X_train, train_labels,
            X_val, val_labels,
            X_test, test_labels,
            tokenizer, vocab_size)
