from flask import Flask, request, jsonify, render_template
from transformers import BertTokenizer, BertForTokenClassification
import torch
import os

app = Flask(__name__)

model_paths = {
    "finetuned_tiny": "./model_tiny",
    "finetuned_mini": "./model_mini",
    "finetuned_small": "./model_small",
    "finetuned_medium": "./model_medium",
    "finetuned_base": "./model_base",
    "pretrained_tiny": "ytu-ce-cosmos/turkish-tiny-bert-uncased",
    "pretrained_mini": "ytu-ce-cosmos/turkish-mini-bert-uncased",
    "pretrained_small": "ytu-ce-cosmos/turkish-small-bert-uncased",
    "pretrained_medium": "ytu-ce-cosmos/turkish-medium-bert-uncased",
    "pretrained_base": "ytu-ce-cosmos/turkish-base-bert-uncased"
}

models = {}
tokenizers = {}

label_map = {
    'UPPER': 1, 'VIRGUL': 2, 'NOKTA': 3, 'IKINOKTA': 4, 'SORU': 5,
    'UPPER_VIRGUL': 6, 'UPPER_NOKTA': 7, 'UPPER_IKINOKTA': 8, 'UPPER_SORU': 9,
    'NONE': 0
}
reverse_label_map = {v: k for k, v in label_map.items()}

def load_model_and_tokenizer(model_key):
    if model_key not in models:
        model_path = model_paths[model_key]
        models[model_key] = BertForTokenClassification.from_pretrained(model_path)
        tokenizers[model_key] = BertTokenizer.from_pretrained(model_path)
    return models[model_key], tokenizers[model_key]

def predict_punctuation(model, tokenizer, text):
    inputs = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=512,
        truncation=True,
        padding='max_length',
        return_tensors='pt'
    )
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

    predictions = torch.argmax(logits, dim=2)
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    predicted_labels = [reverse_label_map[label.item()] for token, label in zip(tokens, predictions[0]) if token not in ['[CLS]', '[SEP]', '[PAD]']]
    tokens = [token for token in tokens if token not in ['[CLS]', '[SEP]', '[PAD]']]

    final_tokens = []
    final_labels = []
    current_token = ""
    current_label = ""

    for token, label in zip(tokens, predicted_labels):
        if token.startswith("##"):
            current_token += token[2:]
        else:
            if current_token:
                final_tokens.append(current_token)
                final_labels.append(current_label)
            current_token = token
            current_label = label

    if current_token:
        final_tokens.append(current_token)
        final_labels.append(current_label)

    return list(zip(final_tokens, final_labels))

def reconstruct_sentence_with_punctuation(predictions):
    sentence = ""
    for token, label in predictions:
        if label == 'NONE':
            sentence += token + " "
        elif label == 'UPPER':
            sentence += token.capitalize() + " "
        elif label == 'VIRGUL':
            sentence += token + ", "
        elif label == 'NOKTA':
            sentence += token + ". "
        elif label == 'IKINOKTA':
            sentence += token + ": "
        elif label == 'SORU':
            sentence += token + "? "
        elif label == 'UPPER_VIRGUL':
            sentence += token.capitalize() + ", "
        elif label == 'UPPER_NOKTA':
            sentence += token.capitalize() + ". "
        elif label == 'UPPER_IKINOKTA':
            sentence += token.capitalize() + ": "
        elif label == 'UPPER_SORU':
            sentence += token.capitalize() + "? "
        else:
            sentence += token + " "

    return sentence.strip()

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data['text']
    model_key = data['model']
    model, tokenizer = load_model_and_tokenizer(model_key)
    predictions = predict_punctuation(model, tokenizer, text)
    reconstructed_sentence = reconstruct_sentence_with_punctuation(predictions)
    return jsonify({
        "predictions": predictions,
        "reconstructed_sentence": reconstructed_sentence
    })

if __name__ == '__main__':
    app.run(debug=True)
