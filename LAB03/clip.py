"""
Custom HuggingFace Trainer and collator for CLIP
"""

# Import external libraries
from transformers import Trainer
import torch
import torch.nn.functional as F

class CLIPTrainer(Trainer):
    """
    Custom huggingface trainer 
    """
    def __init__(self, class_names, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_names = class_names
        self.text_features_cache = None
        
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Custom function to compute contrastive loss for CLIP model
        """
        device = model.device

        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pixel_values=inputs["pixel_values"],
        )

        logits_per_image = outputs.logits_per_image  # shape: (batch_size, batch_size)
        logits_per_text = outputs.logits_per_text

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
        """
        Cache text features for all classes to use in evaluation
        """
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

            # Get image features
            image_features = model.get_image_features(pixel_values=inputs["pixel_values"].to(device))
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # Get text features for all classes
            text_features = self._get_text_features_for_all_classes(model)
            
            # Multiply to get similarity
            similarity = image_features @ text_features.T  # (batch_size, num_classes)
            preds = similarity.argmax(dim=-1)
            
            # Get true labels
            true_labels = inputs["labels"].to(device)

  
            preds = preds.detach().cpu()
            true_labels = true_labels.detach().cpu()

        if prediction_loss_only:
            return (loss, None, None)
        return (loss, preds, true_labels)
    


def CLIPCollator(batch):
    """
    Custom collator for CLIP datasets
    """
    return {
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        "pixel_values": torch.stack([item["pixel_values"] for item in batch]),
        "labels": torch.tensor([item.get("labels", -1) for item in batch], dtype=torch.long)
    }

