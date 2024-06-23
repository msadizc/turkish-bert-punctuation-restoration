import torch
from transformers import BertTokenizer, BertForTokenClassification

tokenizer = BertTokenizer.from_pretrained("./model_base")
model = BertForTokenClassification.from_pretrained("./model_base")

label_map = {
    'UPPER': 1, 'VIRGUL': 2, 'NOKTA': 3, 'IKINOKTA': 4, 'SORU': 5,
    'UPPER_VIRGUL': 6, 'UPPER_NOKTA': 7, 'UPPER_IKINOKTA': 8, 'UPPER_SORU': 9,
    'NONE': 0
}
reverse_label_map = {v: k for k, v in label_map.items()}

def predict_punctuation(text):
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

# Örnek kullanım.
input_text = "küçükleri yüz göz kir pas içinde üst başlarını da öyle bulunca bir tokat da grigoriye aşk etti"
predictions = predict_punctuation(input_text)

# Tahminler ile cümleyi yeniden oluştur.
reconstructed_sentence = reconstruct_sentence_with_punctuation(predictions)
print(reconstructed_sentence)
