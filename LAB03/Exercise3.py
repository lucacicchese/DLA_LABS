from transformers import CLIPProcessor, CLIPModel, TrainingArguments, Trainer
from datasets import load_dataset, get_dataset_split_names
import torch
from PIL import Image
import torch.nn as nn
import random
import wandb
from torch.utils.tensorboard import SummaryWriter
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import default_data_collator
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np



class CLIPTinyImageNetLazyDataset(Dataset):
    def __init__(self, hf_dataset_split, processor, class_names):
        self.dataset = hf_dataset_split
        self.processor = processor
        self.class_names = class_names  # List of class names

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image = sample["image"]
        label = sample["label"]

        text = self.class_names[label]  # Convert label to class name string

        inputs = self.processor(
            text=text,
            images=image,
            return_tensors="pt",
            padding="max_length",
            truncation=True
        )

        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "labels": int(label)
            
        }

def clip_collator(batch):
    """
    Custom collator for CLIP datasets.
    Ensures labels are included and batched properly.
    """
    return {
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        "pixel_values": torch.stack([item["pixel_values"] for item in batch]),
        "labels": torch.tensor([item.get("labels", -1) for item in batch], dtype=torch.long)
    }

class CLIPTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # inputs contains: input_ids, attention_mask, pixel_values
        device = model.device

        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pixel_values=inputs["pixel_values"],
        )

        # Extract logits
        logits_per_image = outputs.logits_per_image  # shape: (batch_size, batch_size)
        logits_per_text = outputs.logits_per_text

        # Create labels for contrastive loss: diagonal elements are positive pairs
        batch_size = logits_per_image.size(0)
        labels = torch.arange(batch_size).to(logits_per_image.device)

        loss_img = F.cross_entropy(logits_per_image, labels)
        loss_txt = F.cross_entropy(logits_per_text, labels)
        loss = (loss_img + loss_txt) / 2

        if return_outputs:
            return loss, {
                "logits": logits_per_image,   
                "labels": inputs["labels"]
            }
        else:
            return loss
        
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        device = model.device

        with torch.no_grad():  # important: disables grad
            outputs = model(
                input_ids=inputs["input_ids"].to(device),
                attention_mask=inputs["attention_mask"].to(device),
                pixel_values=inputs["pixel_values"].to(device),
            )

            # logits
            logits_per_image = outputs.logits_per_image
            batch_size = logits_per_image.size(0)
            labels = torch.arange(batch_size, device=device)

            loss_img = F.cross_entropy(logits_per_image, labels)
            loss_txt = F.cross_entropy(outputs.logits_per_text, labels)
            loss = (loss_img + loss_txt) / 2

            preds = logits_per_image.argmax(dim=-1)

            # detach tensors and ensure 1D
            preds = preds.detach().cpu().view(-1)
            labels = labels.detach().cpu().view(-1)

        if prediction_loss_only:
            return (loss, None, None)
        return (loss, preds, labels)

def preprocess(batch):
    texts = [f"a photo of a {class_names[label]}" for label in batch["label"]]
    images = batch["image"]

    inputs = processor(
        text=texts,
        images=images,
        return_tensors="pt",
        padding=True,
        truncation=True
    )

    return inputs

def get_lazy_preprocess_fn(class_names, processor):
    def lazy_preprocess(example):
        image = example["image"]
        label = example["label"]
        text = f"a photo of a {class_names[label]}"

        inputs = processor(text=text, images=image, return_tensors="pt", padding="max_length", truncation=True)

        example["input_ids"] = inputs["input_ids"].squeeze(0)
        example["attention_mask"] = inputs["attention_mask"].squeeze(0)
        example["pixel_values"] = inputs["pixel_values"].squeeze(0)
        example["labels"] = label 
        return example

    return lazy_preprocess

def lazy_preprocess(example):
    image = example["image"]
    text = example["text"]
    inputs = processor(text=text, images=image, return_tensors="pt", padding="max_length", truncation=True)

    example["input_ids"] = inputs["input_ids"].squeeze(0)
    example["attention_mask"] = inputs["attention_mask"].squeeze(0)
    example["pixel_values"] = inputs["pixel_values"].squeeze(0)
    example["labels"] = 0
    return example

def preprocess_logits_for_metrics(logits, labels):
    
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1) 

def compute_metrics(p):
    print("⚡ compute_metrics called! ⚡")

    preds = p.predictions
    labels = p.label_ids

    # Force 1D arrays, even if scalar
    if np.isscalar(preds):
        preds = np.array([preds])
    else:
        preds = np.atleast_1d(np.array(preds))

    if np.isscalar(labels):
        labels = np.array([labels])
    else:
        labels = np.atleast_1d(np.array(labels))

    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')

    metrics ={
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1)
    }
    print(f"Metrics: {metrics}")

    return metrics

def zero_shot_eval(model, processor, dataset_val):
    class_names = dataset_val['validation'].features['label'].names
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create text inputs for all classes
    templates = [f"a photo of a {label}" for label in class_names]

    # Preprocess all text templates once
    text_inputs = processor(text=templates, return_tensors="pt", padding=True, truncation=True).to(device)

    with torch.no_grad():
        text_features = model.get_text_features(**text_inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    predictions = []
    ground_truths = []

    for example in tqdm(dataset_val["validation"], desc="Zero-shot evaluation"):
        image = example["image"]
        label = example["label"]

        # Preprocess the image
        image_inputs = processor(images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            image_features = model.get_image_features(**image_inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # Compute cosine similarity
            logits_per_image = image_features @ text_features.T
            probs = logits_per_image.softmax(dim=-1)
            prediction = probs.argmax(dim=-1).item()

        predictions.append(prediction)
        ground_truths.append(label)

    accuracy = accuracy_score(ground_truths, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(ground_truths, predictions, average='weighted')

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

    print(f"Zero-shot metrics = {metrics}")

    return metrics


def lora(model, processor, dataset_train, dataset_val, config):


    target_modules = ["q_proj", "v_proj"]


    lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=target_modules,
    lora_dropout=0.1
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir="./clip_lora_finetuned",
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=8,
        num_train_epochs=config["num_epochs"],
        learning_rate=config["learning_rate"],
        logging_dir="./logs",
        logging_steps=config["logging"]["logging_steps"],
        save_strategy="steps",
        eval_strategy="steps",
        eval_steps=100,               
        save_steps=100,
        max_steps=100,
        eval_accumulation_steps=1,
        fp16=True,
        prediction_loss_only=False,
        dataloader_pin_memory=True,
        

    )

    trainer = CLIPTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset_train,
        eval_dataset=dataset_val,
        data_collator=clip_collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
    )

    trainer.train()

    metrics = trainer.evaluate()
    



    return metrics


if __name__ == "__main__":

    config ={
        "project_name": "LAB03_Exercise3",
        "dataset_name_train": "Multimodal-Fatima/TinyImagenet_train",
        "dataset_name_val": "Multimodal-Fatima/TinyImagenet_validation",
        "model_name": "openai/clip-vit-base-patch16",
        "model_max_length": 512,
        "batch_size": 16,
        "num_epochs": 2,
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

    if config["logging"]["wandb"]: 
        wandb.init(project=config["project_name"], config=config)
    if config["logging"]["tensorboard"]:
        writer = SummaryWriter(log_dir=f"logs/{config['logging']['tb_logs']}")

    # Dataset import and model setup
    print(f"Loading dataset: {config['dataset_name_train']}")
    dataset_train = load_dataset(config['dataset_name_train'])
    print(f"Loading dataset: {config['dataset_name_val']}")
    dataset_val = load_dataset(config['dataset_name_val'])

    split_names_train = get_dataset_split_names(config['dataset_name_train'])
    split_names_val = get_dataset_split_names(config['dataset_name_val'])

    print(f"Available splits for {config['dataset_name_train']}: {split_names_train}")
    print(f"Available splits for {config['dataset_name_val']}: {split_names_val}")

    #sprint(dataset_train["train"].features)


    class_names = dataset_train['train'].features['label'].names

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = CLIPModel.from_pretrained(config["model_name"]).to(device)
    processor = CLIPProcessor.from_pretrained(config["model_name"])


    # Zero-shot evaluation
    #zero_shot_metrics = zero_shot_eval(model, processor, dataset_val)
    #print("Zero-shot:", zero_shot_metrics)

    # LoRA
    

    processed_train = CLIPTinyImageNetLazyDataset(dataset_train["train"], processor, class_names)
    processed_val = CLIPTinyImageNetLazyDataset(dataset_val["validation"], processor, class_names)


    lora_metrics = lora(model, processor, processed_train, processed_val, config)
    print("LoRA:", lora_metrics)


