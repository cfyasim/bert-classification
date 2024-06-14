from datasets import Dataset, DatasetDict
import os
import pandas as pd
from time import gmtime, strftime
from tqdm.auto import tqdm
import torch_xla.core.xla_model as xm
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_scheduler


model_name = "bert-base-uncased"
# define xla as device for using AWS Trainium Neuron Cores
device = "xla"

batch_size = 8
num_epochs = 10

print("Device: {}".format(device))


# This method returns a dictionary of input_ids, token_type_ids, attention_mask
def tokenize_and_encode(data):
    results = tokenizer(data["text"], padding="max_length", max_length=512, truncation=True)
    return results


train = pd.read_excel('Global_Reject_Training_Data_Feb22_Encoded.xlsx')

train_dataset = Dataset.from_dict(train)

hg_dataset = DatasetDict({"train": train_dataset})

# Loading Hugging Face AutoTokenizer for the defined model
tokenizer = AutoTokenizer.from_pretrained(model_name)

ds_encoded = hg_dataset.map(tokenize_and_encode, batched=True, remove_columns=["text"])

ds_encoded.set_format("torch")

# Creating a DataLoader object for iterating over it during the training epochs
train_dl = DataLoader(ds_encoded["train"], shuffle=True, batch_size=batch_size)

# Loading Hugging Face pre-trained model for sequence classification for the defined model
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
model.to(device)

current_timestamp = strftime("%Y-%m-%d-%H-%M", gmtime())

no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.2},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.0}
]
learning_rate = 1e-6
adam_epsilon = 1e-8
optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon, weight_decay=1e-5)

num_training_steps = num_epochs * len(train_dl)
progress_bar = tqdm(range(num_training_steps))
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

print("Start training: {}".format(strftime("%Y-%m-%d %H:%M:%S", gmtime())))

# Start model training and defining the training loop
model.train()
for epoch in range(num_epochs):
    for batch in train_dl:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        # xm.mark_step is executing the current graph, updating the model params, and notify end of step to Neuron Core
        xm.mark_step()
        optimizer.zero_grad()
        progress_bar.update(1)

    print("Epoch {}, rank {}, Loss {:0.4f}".format(epoch, xm.get_ordinal(), loss.detach().to("cpu")))

print("End training: {}".format(strftime("%Y-%m-%d %H:%M:%S", gmtime())))

# Using XLA for saving model after training for being sure only one copy of the model is saved
os.makedirs("./models/checkpoints/{}".format(current_timestamp), exist_ok=True)
checkpoint = {"state_dict": model.state_dict()}
xm.save(checkpoint, "./models/checkpoints/{}/checkpoint.pt".format(current_timestamp))
