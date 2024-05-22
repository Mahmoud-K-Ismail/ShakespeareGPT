# generate.py
import numpy as np
from keras.models import load_model
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk  # Import ttk for the combobox

# Load preprocessed data
data = np.load('preprocessed_data.npz', allow_pickle=True)
maximum_seq_length = int(data['maximum_seq_length'])
voc_chars = data['voc_chars']
char_indices = data['char_indices'].item()
indices_char = data['indices_char'].item()

# Load the trained model
model_1 = load_model('model_1.h5')
model_emb_m2m = load_model('model_2.h5')
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

def generate_next(model, text, char_indices, indices_char, maximum_seq_length, num_generated=500):
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

def generate_sample(model, text, char_indices, indices_char, maximum_seq_length, num_generated=500,count = 6, temperature=1.0):
    generated = text
    sentence = text[-maximum_seq_length:]
    c = 0
    while (True):
        x = get_tensor(sentence, maximum_seq_length, char_indices)
        x = x[np.newaxis, :]
        predictions = model.predict(x)[0]
        next_index = sample(predictions, temperature)
        next_char = indices_char[next_index]
        generated += next_char
        sentence = sentence[1:] + next_char
        if next_char in ['\n', '.', '?', '!'] :
           c +=1
           if c == count:
                break
    return generated

def get_tensor_emb(sentence, maximum_seq_length, voc):
    x = np.array([[voc.get(idx, 0) for idx in sentence]], dtype=np.int32)
    return x

def generate_sample_emb(model, text, num_generated=120, temperature=1.0):
    # Initialize generated text with input text.
    generated = text
    # Get last part of input text.
    sentence = text[-maximum_seq_length:]
    # Loop until generated text reaches desired length.
    for i in range(num_generated):
        # Convert sentence to tensor embeddings.
        x = get_tensor_emb(sentence, maximum_seq_length, voc = char_indices)
        # Get model predictions for next character.
        predictions = model.predict(x)[0]
        # Sample next character from predictions.
        next_index = sample(predictions, temperature)
        # Get character corresponding to sampled index.
        next_char =  indices_char[next_index]
        # Add character to generated text.
        generated += next_char
        # Update sentence for next iteration.
        sentence = sentence[1:] + next_char
    # Return generated text.
    return(generated)

def generate_text():
    input_text = entry.get().strip()
    if len(input_text.split()) != 6:
        messagebox.showerror("Error", "Please enter exactly 6 words.")
        return

    sentence_count = int(sentence_count_var.get())

    output_1 = generate_sample(model_1, input_text.lower(), char_indices, indices_char, maximum_seq_length, temperature=0.7, count=sentence_count)
    output_2 = generate_sample_emb(model_emb_m2m, input_text.lower(), temperature = 0.7)
    output_text_1.set("Model 1 Output:\n" + output_1)
    output_text_2.set("Model 2 Output:\n" + output_2)

def reset_fields():
    entry.delete(0, tk.END)
    output_text_1.set("")
    output_text_2.set("")

# Initialize main window
root = tk.Tk()
root.title("Text Generation GUI")

# Initialize variables
output_text_1 = tk.StringVar()
output_text_2 = tk.StringVar()

# Create and place widgets
tk.Label(root, text="Enter a 6-word sentence:").pack(pady=5)
entry = tk.Entry(root, width=50)
entry.pack(pady=5)

# Add a dropdown menu for selecting the number of sentences to generate
tk.Label(root, text="Select number of sentences:").pack(pady=5)
sentence_count_var = tk.StringVar(value='6')
sentence_count_dropdown = ttk.Combobox(root, textvariable=sentence_count_var)
sentence_count_dropdown['values'] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # Add more values if needed
sentence_count_dropdown.pack(pady=5)

tk.Button(root, text="Generate Text", command=generate_text).pack(pady=10)

tk.Label(root, textvariable=output_text_1, wraplength=400, justify="left").pack(pady=5)
tk.Label(root, textvariable=output_text_2, wraplength=400, justify="left").pack(pady=5)

tk.Button(root, text="Generate Another Text", command=reset_fields).pack(pady=10)
tk.Button(root, text="Quit", command=root.quit).pack(pady=10)

# Run the application
root.mainloop()









'''
## BEAM MODEL (Not very successful)
def generate_beam(model, text, char_indices, indices_char, maximum_seq_length, beam_size=5, num_generated=500):
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
    return generated + best_sequence'''
