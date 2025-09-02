
"""
Custom Class to load dataset elements only when needed to save memory
"""
from torch.utils.data import Dataset

class LazyDataset(Dataset):
    """
    This class is used to loaditems from the TinyImageNet dataset
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

