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

#Input verisini hazırlayıp tahmin almak için fonksiyon.
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

    #Gradyan hesaplamalarını devre dışı bırak
    with torch.no_grad():
        #Tahminler alınır.
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

    #Logitleri tahmin edilen tamsayı etiketlere dönüştürür.
    predictions = torch.argmax(logits, dim=2)

    #Padding ve özel tokenleri kaldırır.
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    predicted_labels = [reverse_label_map[label.item()] for token, label in zip(tokens, predictions[0]) if token not in ['[CLS]', '[SEP]', '[PAD]']]
    tokens = [token for token in tokens if token not in ['[CLS]', '[SEP]', '[PAD]']]

    #Ekleri kelime kökleriyle birleştirir ve labelları buna göre ayarlar.
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

#Örnek kullanım. Buraya rastgele bir cümle girilerek tahminler alınabilir.
input_text = "küçükleri yüz göz kir pas içinde üst başlarını da öyle bulunca bir tokat da grigoriye aşk etti"
predictions = predict_punctuation(input_text)

#Tahminler yazdırılır.
for token, label in predictions:
    print(f"Token: {token}, Label: {label}")