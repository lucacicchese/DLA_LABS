"""
LAB03
Exercise 2

Fine-tuning DistilBERT for sentiment analysis
"""

# Import external libraries
from datasets import load_dataset
from transformers import AutoTokenizer, pipeline, AutoModelForSequenceClassification, DataCollatorWithPadding, TrainingArguments, Trainer
import torch
import wandb
from torch.utils.tensorboard import SummaryWriter
from utils import compute_metrics

def tokenize_function(examples):
    
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=config['model_max_length'])

if __name__ == "__main__":
    config ={
        "project_name": "LAB03_Exercise2",
        "dataset_name": "rotten_tomatoes",
        "model_name": "distilbert-base-uncased",
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

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if config["logging"]["wandb"]: 
        wandb.init(project=config['project_name'], config=config)
    if config["logging"]["tensorboard"]:
        writer = SummaryWriter(log_dir=f"logs/{config['logging']['tb_logs']}")

    feature_extractor = pipeline("feature-extraction", model=config['model_name'], framework="pt")

    # Exercise 2.1
    print(f"Loading dataset: {config['dataset_name']}")
    dataset = load_dataset(config['dataset_name'])

    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])

    training_split = dataset['train']
    validation_split = dataset['validation']
    test_split = dataset['test']
    print(f"Training split size: {len(training_split)}, Validation split size: {len(validation_split)}, Test split size: {len(test_split)}")

    print(f"Tokenizing dataset: {config['dataset_name']}")
    tokenized_train = training_split.map(tokenize_function, batched=True)
    tokenized_validation = validation_split.map(tokenize_function, batched=True)
    tokenized_test = test_split.map(tokenize_function, batched=True)

    assert 'input_ids' in tokenized_train.column_names, "Missing input_ids in train dataset!"
    assert 'attention_mask' in tokenized_train.column_names, "Missing attention_mask in train dataset!"
    assert 'input_ids' in tokenized_validation.column_names, "Missing input_ids in validation dataset!"
    assert 'attention_mask' in tokenized_validation.column_names, "Missing attention_mask in validation dataset!"
    assert 'input_ids' in tokenized_test.column_names, "Missing input_ids in test dataset!"
    assert 'attention_mask' in tokenized_test.column_names, "Missing attention_mask in test dataset!"

    train_labels = tokenized_train['label']
    num_classes = len(set(train_labels))

    # Exercise 2.2
    print(f"Loading model for sequence classification with {num_classes} classes...")
    model = AutoModelForSequenceClassification.from_pretrained(config['model_name'], num_labels=num_classes)

    print(model)

    # Exercise 2.3
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True, return_tensors="pt")

    training_args = TrainingArguments(
        output_dir = config['logging']['save_dir'],            
        save_strategy = "steps",
        eval_strategy = "steps",
        eval_steps = 100,
        max_steps = 1000,
        learning_rate = config['learning_rate'],
        per_device_train_batch_size = config['batch_size'],
        per_device_eval_batch_size = config['batch_size'],
        num_train_epochs = config['num_epochs'],
        logging_dir = 'logs/',
        logging_steps = config['logging']['logging_steps'],
        save_steps = config['logging']['save_frequency'],
        save_total_limit = 2,
        load_best_model_at_end=True,
        dataloader_drop_last = True
    )      

    trainer = Trainer(
        model = model,
        args = training_args,
        train_dataset = tokenized_train,
        eval_dataset = tokenized_validation,
        data_collator = data_collator,
        compute_metrics = compute_metrics
    )

    trainer.train()

    results = trainer.evaluate()

    print("Evaluation Results:", results)

    test_results = trainer.evaluate(eval_dataset=tokenized_test)
    print("Test Results:", test_results)


    if config["logging"]["wandb"]:
        wandb.log({"Evaluation results": results})
        wandb.log({"Test results": test_results})
