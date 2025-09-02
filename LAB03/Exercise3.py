"""
LAB03
Exercise 3.2

Fine-tuning CLIP with lora
"""

# Import external libraries
from transformers import CLIPProcessor, CLIPModel
from datasets import load_dataset, get_dataset_split_names
import torch
import wandb
from torch.utils.tensorboard import SummaryWriter


# Import my modules
from clip import CLIPTrainer
from lora import lora
from utils import zero_shot_eval
from dataset import LazyDataset


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

    # Create lazy datasets
    processed_train = LazyDataset(dataset_train["train"], processor, class_names)
    processed_val = LazyDataset(dataset_val["validation"], processor, class_names)

    # LoRA training
    lora_metrics = lora(model, processor, processed_train, processed_val, config, class_names)
    print("LoRA:", lora_metrics)
