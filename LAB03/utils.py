"""
Custom HuggingFace Trainer and collator for CLIP
"""

# Import exernal libraries
from torch.utils.data import DataLoader
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm


def extract_features(model, feature_extractor, dataset, config, device='cpu'):
    """
    Extract features from the dataset using the specified model and feature extractor
    """
    features = []
    labels = []
    print(config['batch_size'])

    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False)

    for batch in dataloader:
        texts = batch['text']
        batch_labels = batch['label']

        output_features = feature_extractor(texts)

        cls_token_vector = []
        for out in output_features:  
            cls_vector = out[0][0]  
            cls_token_vector.append(cls_vector)

        
        features.extend(cls_token_vector)
        labels.extend(batch_labels)

    return np.stack(features), np.array(labels)

def compute_metrics(p):
    """
    Compute metrics for model evaluation
    """
    logits = p.predictions
    labels = p.label_ids

    predictions = torch.tensor(logits).argmax(dim=-1).numpy()

    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def zero_shot_eval(model, processor, dataset_val):
    """
    Evaluate the model performance before fine-tuning
    """
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

        image_inputs = processor(images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            image_features = model.get_image_features(**image_inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            
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