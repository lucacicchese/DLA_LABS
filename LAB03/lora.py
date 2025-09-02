"""
Trainer for fine-tuning CLIP with LoRA
"""

# Import external libraries
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments
import torch
import numpy as np

# Import my modules
from clip import CLIPCollator, CLIPTrainer

def preprocess_logits_for_metrics(logits, labels):
    """
    This function is called by the trainer to preprocess logits before computing metrics
    """

    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1) 


def compute_metrics(p):
    """
    Compute classification metrics
    """

    preds = p.predictions
    labels = p.label_ids
        
    preds = np.atleast_1d(np.array(preds)).flatten()
    labels = np.atleast_1d(np.array(labels)).flatten()

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

def lora(model, processor, dataset_train, dataset_val, config, class_names):
    """
    This function fine-tunes and evaluates the CLIP model using lora
    """

    # These are the modules that will be fine tuned using Lora,
    # q_proj and v_proj have been chosen as they have proven to be the most effective
    # considering the amount of parameters they have
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
        eval_steps=1000,               
        save_steps=1000,
        max_steps=10000,
        eval_accumulation_steps=1,
        fp16=True,
        prediction_loss_only=False,
        dataloader_pin_memory=True,
        remove_unused_columns=False,  
    )

    trainer = CLIPTrainer(
        class_names=class_names,  
        model=model,
        args=training_args,
        train_dataset=dataset_train,
        eval_dataset=dataset_val,
        data_collator=CLIPCollator,
        compute_metrics=compute_metrics,
        processing_class=processor,  
    )

    trainer.train()
    metrics = trainer.evaluate()
    return metrics