"""
Function that generates Class Activation Maps (CAM) for a given input image and model.
"""

# Import external libraries
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import os
from collections import defaultdict

# Import my modules


def generate_class_activation_map(model, input_image, target_class=None, original_size=None):

    model.eval()
    
    with torch.no_grad():
        output = model(input_image)

    feature_maps = model.get_last_conv_output()  # Shape: (1, num_classes, H, W)
    

    if target_class is None:
        predicted_class = output.argmax(dim=1).item()
    else:
        predicted_class = target_class
    
    # Get confidence score
    probabilities = F.softmax(output, dim=1)
    confidence = probabilities[0, predicted_class].item()
    
    # Generate CAM for the target class
    cam = feature_maps[0, predicted_class].cpu().numpy()  
    
    cam = np.maximum(cam, 0)
    
    if cam.max() > 0:
        cam = (cam - cam.min()) / (cam.max() - cam.min())
    
    if original_size is not None:
        cam = F.interpolate(
            torch.tensor(cam).unsqueeze(0).unsqueeze(0), 
            size=original_size, 
            mode='bilinear', 
            align_corners=False
        ).squeeze().numpy()
    
    return cam, predicted_class, confidence


def save_cam_visualization(original_image, cam, predicted_class, confidence, true_label, class_names, save_path, filename):

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    

    img_display = np.transpose(original_image, (1, 2, 0))
    img_display = img_display * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    img_display = np.clip(img_display, 0, 1)
    axes[0].imshow(img_display)

    axes[0].set_title('Original Image')
    axes[0].axis('off')
    

    axes[1].imshow(cam, cmap='jet')
    axes[1].set_title('Class Activation Map')
    axes[1].axis('off')
    
    # Overlay
    axes[2].imshow(img_display)
    
    axes[2].imshow(cam, cmap='jet', alpha=0.4)
    axes[2].set_title('CAM Overlay')
    axes[2].axis('off')
    
    # Add text with prediction information
    pred_name = class_names[predicted_class] 
    true_name = class_names[true_label] 
    
    fig.suptitle(f'Predicted: {pred_name} ({confidence:.3f}) | True: {true_name}', 
                 fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, filename), dpi=150, bbox_inches='tight')
    plt.close()


def cam(model, test_data, device, num_samples_per_class=3, save_folder='cam_results'):

    os.makedirs(save_folder, exist_ok=True)

    class_names = test_data.dataset.class_names

    model.eval()
    
    # Group samples by class
    samples_by_class = defaultdict(list)
    
    # Get n samples for each class
    for idx, (image, label) in enumerate(test_data.dataset):
        if len(samples_by_class[label]) < num_samples_per_class:
            samples_by_class[label].append((idx, image, label))
        if len(samples_by_class) == 10 and all(len(samples) >= num_samples_per_class for samples in samples_by_class.values()):
            break
    

    for class_idx, samples in samples_by_class.items():
        class_name = class_names[class_idx]
        print(f"\nProcessing class {class_idx} ({class_name})...")

        class_folder = os.path.join(save_folder, f'class_{class_idx}_{class_name}')
        os.makedirs(class_folder, exist_ok=True)
        
        for sample_idx, (dataset_idx, image, true_label) in enumerate(samples):
            # Prepare image for model
            input_image = image.unsqueeze(0).to(device)
            original_image = image.cpu().numpy()
            
            # Generate CAM
            cam, predicted_class, confidence = generate_class_activation_map(
                model, input_image, original_size=image.shape[1:]
            )
            
            filename = f'sample_{sample_idx}_idx_{dataset_idx}.png'
            save_cam_visualization(
                original_image, cam, predicted_class, confidence,
                true_label, class_names, class_folder, filename
            )
