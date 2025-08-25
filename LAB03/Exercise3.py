from transformers import CLIPProcessor, CLIPModel, TrainingArguments, Trainer
from datasets import load_dataset
import torch
from PIL import Image
import torch.nn as nn
import random

# Define the compute_metrics function to evaluate accuracy
def compute_metrics(p):
    logits, labels = p
    predictions = torch.argmax(logits, axis=-1)
    accuracy = (predictions == labels).sum().item() / labels.size(0)
    return {"accuracy": accuracy}



config ={
    "project_name": "LAB03_Exercise3",
    "dataset_name": "zh-plus/tiny-imagenet",
    "model_name": "openai/clip-vit-base-patch16",
    "model_max_length": 512,
    "batch_size": 16,
    "num_epochs": 5,
    "learning_rate": 2e-5,

    "logging": {
        "tensorboard": True,
        "weightsandbiases": True,
        "wandb": True,  
        "tb_logs": "tensorboard_runs",  
        "save_dir": "checkpoints",      
        "save_frequency": 100,
        "logging_steps": 10
    }
}


model_name = config['model_name']
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)


dataset = load_dataset(config['dataset_name'])

#print(dataset)


random_index = random.randint(0, len(dataset["train"]) - 1)

# Get the image and label at that index
image = dataset["train"][random_index]["image"]
label = dataset["train"][random_index]["label"]



# Set of labels for zero-shot classification
labels = ["a photo of a cat", "a photo of a dog", "a photo of a horse", "a photo of a bird"]


inputs = processor(text=labels, images=image, return_tensors="pt", padding=True)

# Perform zero-shot classification
with torch.no_grad():
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image  
    probs = logits_per_image.softmax(dim=1)  


predicted_class = torch.argmax(probs, dim=1)
print(f"Predicted label: {labels[predicted_class]}")


for param in model.text_model.parameters():
    param.requires_grad = False


image_encoder = model.vision_model
num_labels = len(dataset["train"].unique("label"))  

classification_head = nn.Sequential(
    nn.Linear(image_encoder.config.hidden_size, 512),
    nn.ReLU(),
    nn.Linear(512, num_labels)
)
model.classification_head = classification_head

print(model)


def compute_loss(model, inputs, labels):

    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    loss_fct = nn.CrossEntropyLoss()
    loss = loss_fct(logits_per_image, labels)
    return loss


training_args = TrainingArguments(
    output_dir="./results",          # output directory for model checkpoints
    evaluation_strategy="epoch",     # evaluate every epoch
    learning_rate=2e-5,              # learning rate
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=16,   # batch size per device during evaluation
    num_train_epochs=3,              # number of training epochs
    weight_decay=0.01,               # weight decay to avoid overfitting
    logging_dir="./logs",            # directory for storing logs
    logging_steps=10,                # log every 10 steps
)


trainer = Trainer(
    model=model,                      # the model to fine-tune
    args=training_args,               # training arguments
    train_dataset=dataset["train"],   # training dataset
    eval_dataset=dataset["valid"],    # evaluation dataset
    compute_metrics=compute_metrics,  # evaluation metrics
    compute_loss=compute_loss,        # use the custom loss function
)


trainer.train()

results = trainer.evaluate()
print("Evaluation Results:", results)
