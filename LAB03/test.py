# Import necessary libraries
from transformers import CLIPProcessor, CLIPModel, TrainingArguments, Trainer, default_data_collator
from datasets import load_dataset, get_dataset_split_names
import torch
import wandb
import numpy as np
from tqdm import tqdm
from peft import LoraConfig, get_peft_model
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter


class LazyDataset(Dataset):
    """
    This class is used  items from the TinyImageNet
    """
    def __init__(self, hf_dataset_split, processor, class_names):
        self.dataset = hf_dataset_split
        self.processor = processor
        self.class_names = class_names 

    def __len__(self):
        """Return dataset lenght"""
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image = sample["image"]
        label = sample["label"]

        text = f"a photo of a {self.class_names[label]}"  # Convert label to class name string

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
    Custom collator for CLIP datasets
    Ensures labels are included and batched properly.
    """
    return {
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        "pixel_values": torch.stack([item["pixel_values"] for item in batch]),
        "labels": torch.tensor([item.get("labels", -1) for item in batch], dtype=torch.long)
    }


class CLIPTrainer(Trainer):
    """Custom huggingface trainer """
    def __init__(self, class_names, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_names = class_names
        self.text_features_cache = None
        
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
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
        contrastive_labels = torch.arange(batch_size, device=device)

        loss_img = F.cross_entropy(logits_per_image, contrastive_labels)
        loss_txt = F.cross_entropy(logits_per_text, contrastive_labels)
        loss = (loss_img + loss_txt) / 2

        if return_outputs:
            return loss, {
                "logits": logits_per_image,   
                "labels": inputs["labels"]
            }
        else:
            return loss
    
    def _get_text_features_for_all_classes(self, model):
        """Cache text features for all classes to use in evaluation"""
        if self.text_features_cache is None:
            device = model.device
            templates = [f"a photo of a {class_name}" for class_name in self.class_names]
            
            text_inputs = self.tokenizer(
                text=templates, 
                return_tensors="pt", 
                padding=True, 
                truncation=True
            ).to(device)
            
            with torch.no_grad():
                text_features = model.get_text_features(**text_inputs)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                self.text_features_cache = text_features
                
        return self.text_features_cache
        
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        device = model.device
        
        with torch.no_grad():
            # Compute contrastive loss (for logging purposes)
            outputs = model(
                input_ids=inputs["input_ids"].to(device),
                attention_mask=inputs["attention_mask"].to(device),
                pixel_values=inputs["pixel_values"].to(device),
            )

            logits_per_image = outputs.logits_per_image
            batch_size = logits_per_image.size(0)
            contrastive_labels = torch.arange(batch_size, device=device)

            loss_img = F.cross_entropy(logits_per_image, contrastive_labels)
            loss_txt = F.cross_entropy(outputs.logits_per_text, contrastive_labels)
            loss = (loss_img + loss_txt) / 2

            # For metrics: do zero-shot classification
            # Get image features
            image_features = model.get_image_features(pixel_values=inputs["pixel_values"].to(device))
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # Get text features for all classes
            text_features = self._get_text_features_for_all_classes(model)
            
            # Compute similarity and get predictions
            similarity = image_features @ text_features.T  # (batch_size, num_classes)
            preds = similarity.argmax(dim=-1)
            
            # Get true labels
            true_labels = inputs["labels"].to(device)

            # Convert to CPU and ensure proper format
            preds = preds.detach().cpu()
            true_labels = true_labels.detach().cpu()

        if prediction_loss_only:
            return (loss, None, None)
        return (loss, preds, true_labels)


def preprocess_logits_for_metrics(logits, labels):
    """This function is called by the trainer to preprocess logits before computing metrics"""
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1) 


def compute_metrics(p):
    """Compute classification metrics"""

    preds = p.predictions
    labels = p.label_ids

    # Ensure arrays are 1D
    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
        
    preds = np.atleast_1d(np.array(preds)).flatten()
    labels = np.atleast_1d(np.array(labels)).flatten()

    # Remove any invalid predictions/labels
    valid_mask = (labels >= 0) & (preds >= 0)
    preds = preds[valid_mask]
    labels = labels[valid_mask]

    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted', zero_division=0)

    metrics = {
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


def lora(model, processor, dataset_train, dataset_val, config, class_names):
    """This function trains and evaluates the newly trained model"""

    # Thise are the modules that will be fine tuned usig Lora, 
    # q_proj and v_proj have been chosen as from 
    target_modules = ["q_proj", "v_proj"]

    # 
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
        eval_steps=1000,               
        save_steps=1000,
        max_steps=10000,
        eval_accumulation_steps=1,
        fp16=True,
        prediction_loss_only=False,
        dataloader_pin_memory=True,
        remove_unused_columns=False,  # Important: keep all columns
    )

    trainer = CLIPTrainer(
        class_names=class_names,  
        model=model,
        args=training_args,
        train_dataset=dataset_train,
        eval_dataset=dataset_val,
        data_collator=clip_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor,  # Pass processor as tokenizer
    )

    trainer.train()
    metrics = trainer.evaluate()
    return metrics


if __name__ == "__main__":
    config = {
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

    class_names = dataset_train['train'].features['label'].names
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = CLIPModel.from_pretrained(config["model_name"]).to(device)
    processor = CLIPProcessor.from_pretrained(config["model_name"])

    # Zero-shot evaluation
    zero_shot_metrics = zero_shot_eval(model, processor, dataset_val)
    print("Zero-shot:", zero_shot_metrics)

    # Create datasets
    processed_train = LazyDataset(dataset_train["train"], processor, class_names)
    processed_val = LazyDataset(dataset_val["validation"], processor, class_names)

    # LoRA training
    lora_metrics = lora(model, processor, processed_train, processed_val, config, class_names)
    print("LoRA:", lora_metrics)
