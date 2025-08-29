
from torch.utils.data import DataLoader
import numpy as np
from transformers import AutoTokenizer
import torch
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support


def extract_features(model, feature_extractor, dataset, config, device='cpu'):
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