# BERT Masked Language Model with Attention Visualization

It allows you to predict masked tokens in a text using **BERT** and visualize the **self-attention scores** of the model.


## Features

- Predicts top K replacements for a masked token in a sentence.
- Visualizes attention weights for each layer and head in the BERT model.
- Generates diagrams where tokens and their attention scores are represented as a grid.


## Usage
```
python main.py
```
When prompted, enter a sentence containing the mask token "[MASK]"
The script will display the top K predictions for the masked token in the console.