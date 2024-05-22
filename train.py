# train.py
import numpy as np
import os
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, LSTM
from keras.callbacks import LambdaCallback, EarlyStopping
from keras.layers import Embedding

# Load preprocessed data
data = np.load('preprocessed_data.npz',allow_pickle=True)
X = data['X']
y = data['y']
maximum_seq_length = int(data['maximum_seq_length'])
voc_chars = data['voc_chars']
char_indices = data['char_indices'].item()
indices_char = data['indices_char'].item()
X_emb = data['X_emb']
y_emb = data['y_emb']

def get_tensor_emb(sentence, maximum_seq_length, voc):
    # Convert sentence to numerical representation using vocab.
    x = np.array([[voc.get(idx, 0) for idx in sentence]], dtype=np.int32)
    # Return numerical representation as tensor.
    return x

# Function to convert sentence to tensor
def get_tensor(sentence, maximum_seq_length, voc):
    # Initialize tensor with zeros
    x = np.zeros((maximum_seq_length, len(voc)), dtype=np.float32)

    # Iterate over characters in sentence
    for i, char in enumerate(sentence):
        # Break if maximum sequence length is reached
        if i >= maximum_seq_length:
            break

        # Check if character is in vocabulary
        if char in voc:
            # Set corresponding value in tensor to 1
            x[i, voc[char]] = 1
        else:
            # Print warning if character not in vocabulary
            print(f"Warning: Character '{char}' not in vocabulary.")

    # Return tensor
    return x

# Define and compile the model
model_1 = Sequential()
model_1.add(LSTM(128, input_shape=(maximum_seq_length, len(voc_chars))))
model_1.add(Dense(len(voc_chars)))
model_1.add(Activation('softmax'))
model_1.compile(loss='categorical_crossentropy', optimizer='adam')

# Function to generate text using the model
def generate_next(model, text, char_indices, indices_char, maximum_seq_length, num_generated=120):
    # Initialize generated text with input text
    generated = text
    # Get last part of input text
    sentence = text[-maximum_seq_length:]
    for i in range(num_generated):
        # Convert sentence to tensor
        x = get_tensor(sentence, maximum_seq_length, char_indices)
        # Add batch dimension
        x = x[np.newaxis, :]
        # Get model predictions
        predictions = model.predict(x)[0]
        # Get index of most likely character
        next_index = np.argmax(predictions)
        # Get character from index
        next_char = indices_char[next_index]
        # Add character to generated text
        generated += next_char
        # Update sentence for next iteration
        sentence = sentence[1:] + next_char
        # Check for sentence ending
        if next_char in ['\n', '.', '?', '!'] and len(generated) > len(text):
            break
    # Return generated text
    return generated

# Function to generate text after each epoch
def end_epoch_generate(epoch, _):
    # Print message after each epoch
    print(f'Generating text after epoch: {epoch + 1}')
    # Sample texts to generate next sequence
    texts_ex = ["From fairest creatures we desire increase,"]
    # Iterate over each sample text
    for text in texts_ex:
        # Generate next sequence using model
        sample = generate_next(model_1, text.lower(), char_indices, indices_char, maximum_seq_length)
        # Print generated sequence
        print(f'{sample}')

# Function to generate text using the model
def generate_next_emb(model, text, num_generated=120):
    # Initialize generated text with input text
    generated = text
    # Get last part of input text (max sequence length)
    sentence = text[-maximum_seq_length:]
    # Convert characters to indices
    char_idxs = [[char_indices[char] for char in sentence]]
    # Generate next characters in sequence
    for i in range(num_generated):
        # Convert indices to array
        x = np.array(char_idxs)
        # Predict next character probabilities
        predictions = model.predict(x)[0]
        # Get index of most likely character
        next_index = np.argmax(predictions)
        # Convert index to character
        next_char = indices_char[next_index]
        # Add character to generated text
        generated += next_char
        # Update indices for next iteration
        char_idxs = [char_idxs[0][1:] + [next_index]]
    # Return generated text
    return(generated)

# Function to generate text after each epoch
def end_epoch_generate2(epoch, _):
    # Print the epoch number  
    print('\n Generating text after epoch: %d' % (epoch+1))
    # Define a list of example texts
    texts_ex = ["From fairest creatures we desire increase,"]
    # Iterate through the example texts
    for text in texts_ex:
        # Generate the next embedding using the model  
        sample = generate_next_emb(model_emb_m2m, text)
        # Print the generated text  
        print('%s' % (sample))  

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
        print("Model 1 saved successfully")
    except Exception as e:
        print(f"Error saving the model: {e}")

print("Training Model 1 done.")

model_emb_m2m = Sequential()
model_emb_m2m.add(Embedding(input_dim=len(voc_chars), output_dim=32, input_length=maximum_seq_length))
model_emb_m2m.add(LSTM(128, input_shape=(maximum_seq_length, 32), return_sequences=False))
model_emb_m2m.add(Dense(len(voc_chars)))
model_emb_m2m.add(Activation('softmax'))

model_emb_m2m.compile(loss='sparse_categorical_crossentropy', optimizer='adam')

# File paths for saving models
model_2_path = 'model_2.h5'

# Check if models are already trained and saved,if not train and save
if os.path.exists(model_2_path):
    print("Model 2 is being imported")
    model_emb_m2m = load_model(model_2_path)
else:
    print("Model 2 is being trained")
    model_emb_m2m.fit(X_emb, y_emb,
                  batch_size=128,
                  epochs=5,
                  validation_split = 0.2,
                  callbacks=[LambdaCallback(on_epoch_end=end_epoch_generate2)])
    try:
        model_emb_m2m.save(model_2_path)
        print("Model 2 saved successfully")
    except Exception as e:
        print(f"Error saving the model: {e}")

print("Training 2 done.")
