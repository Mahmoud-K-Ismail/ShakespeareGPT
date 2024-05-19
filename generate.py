# generate.py
import numpy as np
from keras.models import load_model

# Load preprocessed data
data = np.load('preprocessed_data.npz', allow_pickle=True)
maximum_seq_length = int(data['maximum_seq_length'])
voc_chars = data['voc_chars']
char_indices = data['char_indices'].item()
indices_char = data['indices_char'].item()

# Load the trained model
model_1 = load_model('model_1.h5')

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

def sample(predictions, temperature):
    predictions = np.asarray(predictions).astype('float64')
    log_predictions = np.log(predictions) / temperature
    predictions = np.exp(log_predictions)
    predictions = predictions / np.sum(predictions)
    probas = np.random.multinomial(1, predictions, 1)
    return np.argmax(probas)

def generate_sample(model, text, char_indices, indices_char, maximum_seq_length, num_generated=120, temperature=1.0):
    generated = text
    sentence = text[-maximum_seq_length:]
    for i in range(num_generated):
        x = get_tensor(sentence, maximum_seq_length, char_indices)
        x = x[np.newaxis, :]
        predictions = model.predict(x)[0]
        next_index = sample(predictions, temperature)
        next_char = indices_char[next_index]
        generated += next_char
        sentence = sentence[1:] + next_char
        if next_char in ['\n', '.', '?', '!'] and len(generated) > len(text):
            break
    return generated

def generate_beam(model, text, char_indices, indices_char, maximum_seq_length, beam_size=5, num_generated=120):
    generated = text
    sentence = text[-maximum_seq_length:]
    current_beam = [(0, [], sentence)]
    for l in range(num_generated):
        all_beams = []
        for prob, current_preds, current_input in current_beam:
            x = get_tensor(current_input, maximum_seq_length, char_indices)
            x = x[np.newaxis, :]
            predictions = model.predict(x)[0]
            top_indices = np.argsort(predictions)[-beam_size:]
            possible_next_chars = [indices_char[idx] for idx in top_indices]
            all_beams += [
                (prob + np.log(predictions[idx]),
                 current_preds + [idx],
                 current_input[1:] + indices_char[idx])
                for idx in top_indices]
        current_beam = sorted(all_beams, key=lambda x: x[0], reverse=True)[:beam_size]
    best_beam = max(current_beam, key=lambda x: x[0])
    best_sequence = best_beam[2]
    return generated + best_sequence

# Main loop to prompt user for input and generate text
while True:
    input_text = input("Enter a 6-word sentence: ").strip()
    if len(input_text.split()) != 6:
        print("Please enter exactly 6 words.")
        continue

    output_1 = generate_sample(model_1, input_text.lower(), char_indices, indices_char, maximum_seq_length, temperature=0.7)
    output_2 = generate_beam(model_1, input_text.lower(), char_indices, indices_char, maximum_seq_length)

    print("\nModel 1 Output:\n" + output_1)
    print("\nModel 2 Output:\n" + output_2)

    cont = input("Do you want to generate another text? (yes/no): ").strip().lower()
    if cont != 'yes':
        break

print("Generation done.")