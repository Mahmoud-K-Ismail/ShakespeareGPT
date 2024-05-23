# ShakespeareGPT

ShakespeareGPT is a text generation model designed to emulate the writing style of William Shakespeare. This project leverages deep learning techniques, specifically using Long Short-Term Memory (LSTM) networks, to generate Shakespearean-style text. The model is trained on the complete works of William Shakespeare and can produce coherent paragraphs that mimic the bard's unique style and vocabulary.

## Features

- **Deep Learning Model**: Utilizes an LSTM network to capture the sequential nature of Shakespeare's writing.
- **Text Generation**: Capable of generating text that mimics the style and tone of Shakespeare.
- **Customizable Generation**: Allows users to control the length and structure of the generated text, including sentence count and temperature for sampling.
- **Beam Search**: Implements beam search algorithm for generating more coherent and contextually accurate text.
- **Interactive GUI**: Includes a user-friendly graphical interface for inputting prompts and displaying generated text.

## Models:

- **model_1:** will be referred to as the "Index-Based Model".
- **model_emb_m2m:** will be referred to as the "One-Hot-Encoding-Based Model".


## Getting Started

### Prerequisites

- Python 3.8 or higher
- TensorFlow
- Keras
- NLTK
- NumPy
- Matplotlib

### Installation (Still ongoing)

1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/ShakespeareGPT.git
   cd ShakespeareGPT
   ```
2.Install the required packages:
   ```sh
   pip install -r requirements.txt
   ```
3. Download the dataset: The model is trained on the complete works of William Shakespeare from Project Gutenberg. This script will automatically download and preprocess the data.

### Training the Model
   1. Initialize the dataset:
   ```sh
      python preprocess.py
   ```
   2. Train the model:
   ``` sh
      python train.py
   ```
   3. Save the trained model: The model will be saved as model.h5 in the project directory.

### Generating Text
   1. Run the interactive script:

      ```sh
         python generate.py
      ```
   2. Enter a 6-word sentence as a prompt and the model will generate a Shakespearean paragraph based on it.

### Usage
You can use the interactive script to generate text or integrate the model into other applications. The GUI allows for easy input of prompts and viewing of generated text.

## Example
   ```python
      input_text = "From fairest creatures we desire increase,"
      output = generate_sample(model, input_text.lower(), num_sentences=4, temperature=0.7)
      print(output)
   ```
### Contributing
We welcome contributions! Please read our Contributing Guidelines for more information on how to contribute to this project.

### Acknowledgments
Project Gutenberg for providing the dataset.
Keras and TensorFlow for the deep learning framework.
NLTK for natural language processing tools.
