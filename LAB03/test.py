from datasets import load_dataset
from transformers import CLIPProcessor

dataset = load_dataset("Multimodal-Fatima/TinyImagenet_train")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
class_names = dataset["train"].features["label"].names

def get_lazy_preprocess_fn(class_names, processor):
    def lazy_preprocess(example):
        image = example["image"]
        label = example["label"]
        text = f"a photo of a {class_names[label]}"
        inputs = processor(text=text, images=image, return_tensors="pt", padding="max_length", truncation=True)
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "labels": label
        }
    return lazy_preprocess

# Apply transform correctly
train_dataset = dataset["train"]
train_dataset.set_transform(get_lazy_preprocess_fn(class_names, processor))

# Trigger the transform
print(train_dataset[0])
