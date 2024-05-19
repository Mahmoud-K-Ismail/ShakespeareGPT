# train.py
import numpy as np
import os
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, LSTM
from keras.callbacks import LambdaCallback, EarlyStopping

# Load preprocessed data
data = np.load('preprocessed_data.npz',allow_pickle=True)
X = data['X']
y = data['y']
maximum_seq_length = int(data['maximum_seq_length'])
voc_chars = data['voc_chars']
char_indices = data['char_indices'].item()
indices_char = data['indices_char'].item()

def get_tensor(sentence, maximum_seq_length, voc):
    x = np.zeros((maximum_seq_length, len(voc)), dtype=np.float32)
    for i, char in enumerate(sentence):
        if i >= maximum_seq_length:
            break
        if char in voc:
            x[i, voc[char]] = 1
        else:
            print(f"Warning: Character '{char}' not in vocabulary.")
    return x

# Define and compile the model
model_1 = Sequential()
model_1.add(LSTM(128, input_shape=(maximum_seq_length, len(voc_chars))))
model_1.add(Dense(len(voc_chars)))
model_1.add(Activation('softmax'))
model_1.compile(loss='categorical_crossentropy', optimizer='adam')

# Function to generate text using the model
def generate_next(model, text, char_indices, indices_char, maximum_seq_length, num_generated=120):
    generated = text
    sentence = text[-maximum_seq_length:]
    for i in range(num_generated):
        x = get_tensor(sentence, maximum_seq_length, char_indices)
        x = x[np.newaxis, :]
        predictions = model.predict(x)[0]
        next_index = np.argmax(predictions)
        next_char = indices_char[next_index]
        generated += next_char
        sentence = sentence[1:] + next_char
        if next_char in ['\n', '.', '?', '!'] and len(generated) > len(text):
            break
    return generated

# Function to generate text after each epoch
def end_epoch_generate(epoch, _):
    print(f'Generating text after epoch: {epoch + 1}')
    texts_ex = ["From fairest creatures we desire increase,"]
    for text in texts_ex:
        sample = generate_next(model_1, text.lower(), char_indices, indices_char, maximum_seq_length)
        print(f'{sample}')

# Early stopping callback
early_stopping = EarlyStopping(
    monitor="val_loss",
    min_delta=0,
    patience=3,
    verbose=0,
    mode="auto",
    baseline=None,
    restore_best_weights=True
)

# File paths for saving models
model_1_path = 'model_1.h5'

# Check if models are already trained and saved
if os.path.exists(model_1_path):
    print("Model is being imported")
    model_1 = load_model(model_1_path)
else:
    print("Model is being trained")
    model_1.fit(X, y,
                batch_size=128,
                epochs=10,
                validation_split=0.2,
                callbacks=[LambdaCallback(on_epoch_end=end_epoch_generate), early_stopping])
    try:
        model_1.save(model_1_path)
        print("Model saved successfully")
    except Exception as e:
        print(f"Error saving the model: {e}")

print("Training done.")
