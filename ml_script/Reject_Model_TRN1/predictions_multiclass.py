import pandas as pd
import torch
import torch.neuron
from transformers import AutoTokenizer
import numpy as np
from sklearn.metrics import classification_report


neuron_model = torch.jit.load("global_reject_bert_neuron.pt")  # model saved for multiclass classification
df = pd.read_excel('Global_Reject_Testing_Data_Feb22_Encoded.xlsx')
max_length = 512
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
threshold = 0.85 

def predict(row):
    encoding = tokenizer.encode_plus(
        row,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]
    token_type_ids = encoding["token_type_ids"]
    logits = neuron_model(input_ids, attention_mask, token_type_ids)[0]

    # Apply softmax to get probabilities
    probabilities = torch.nn.functional.softmax(logits, dim=1)

    # Check if the probability of any class is greater than the rest of classes.
    predicted_class = torch.argmax(probabilities).item()

    return predicted_class


df['predicted_label'] = df['text'].apply(predict)

actual_labels = df['labels'].to_list()
predicted_labels = df['predicted_label'].to_list()

print(classification_report(actual_labels, predicted_labels))







