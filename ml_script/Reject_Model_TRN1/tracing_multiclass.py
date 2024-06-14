import torch
import torch_xla.core.xla_model as xm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.neuron

model_name = "bert-base-uncased"
model_path = "/home/mlModelTraining/ml_training/models/checkpoints/2024-02-22-05-44/checkpoint.pt"
device = 'xla'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3, return_dict=False)
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['state_dict'])


example = """Multi splits with water heating. [Source: Climate Control News] Midea has created a multi split system which combines heating, cooling, water heating and heat recovery. The all year round heat pump solution was a joint project between Paolo Lorini, head of Midea RAC Design Milan (MRDM) and Matteo Nunziati, a top Italian interior designer. Nunziati is considered one of the most talented interior designers of his generation. For Nunziati, a holistic design takes into account not only sustainability but also functionality and aesthetics. This has been incorporated into Midea's new multi splits system with integrated water heating. Lorini said most homes today are installed with several independent systems, with an air conditioner for the summer, floor heating for the winter, and a gas boiler for all-year water heating. "This is inefficient, costs too much to run and maintain, and takes up a lot of home space," he said. "Midea has created something completely different: we have integrated all the above systems into a multi-spilt solution that supports up to four indoor air conditioning units, capable of cooling and heating the space, and a water heating module, all connected to a single outdoor unit with heat recovery technology." In summer, free domestic hot water is available when the air-conditioner is running, as its energy can be recovered and reused for water heating. The system comes in two options for households with different priorities. The model equipped with a hydro unit is designed for users who need both the space heating and hot water supply at no additional cost, while one has a coil water tank to allow for fast air cooling and heating while providing hot water the whole year."""
max_length = 512

# Convert sample inputs to a format that is compatible with TorchScript tracing
encoding = tokenizer.encode_plus(
    example,
    max_length=max_length,
    padding="max_length",
    truncation=True,
    return_tensors="pt",
)
input_example = (
    encoding["input_ids"],
    encoding["attention_mask"],
    encoding["token_type_ids"],
)
logits = model(*input_example)
logits[0][0].argmax()

try:
    traced_model = torch.jit.trace(model, input_example)
    print("Cool! Model is jit traceable")
except Exception as e:
    print("Ops. Something went wrong. Model is not traceable")
    print(e)

torch.neuron.analyze_model(model, input_example)
neuron_model = torch.neuron.trace(model, input_example)
neuron_model.save(f"global_reject_bert_neuron.pt")
