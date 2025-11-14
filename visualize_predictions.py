import os
import yaml
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from model import UNet

# ============================================================================
# CONFIG
# ============================================================================
CHECKPOINT_PATH = "/media/data/divyanshu/spleen/code/unet/checkpoints2/onlyspleencases_spleen_best_model_20_val_loss_0.6184_val_dice_0.9113_val_iou_0.8370.pt"
CSV_PATH = "/media/data/divyanshu/spleen/curated_data/annotated_data/kaggle_spleenonly/kaggle_data_onlyspleencases.csv"
IMAGES_DIR = "/media/data/divyanshu/spleen/curated_data/annotated_data/kaggle_spleenonly/images"
MASKS_DIR = "/media/data/divyanshu/spleen/curated_data/annotated_data/kaggle_spleenonly/binary"
CONFIG_PATH = "./config.yaml"
NUM_IMAGES = 6  # imags to viz 
OUTPUT_PNG_PATH = "./visualization_predictions_spleenonlytrained_on_kagglespleencases.png"
IMG_SIZE = 256  
THRESHOLD = 0.5  
# ============================================================================

def load_model(checkpoint_path, config_path):
    """Load model from checkpoint"""
   
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # spin up model
    model = UNet(
        n_channels=config['model']['n_channels'],
        n_classes=config['model']['n_classes'],
        encoder_name=config['model'].get('encoder_name', 'resnet34'),
        encoder_weights=config['model'].get('encoder_weights', 'imagenet')
    )
    
    # Load checkpoint
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Loaded model from: {checkpoint_path}")
    
    
    return model, device, config

def load_and_preprocess_image(img_path, img_size):
    """Load and preprocess a single image for inference"""
    
    img = Image.open(img_path).convert('L')
    img_array = np.array(img, dtype=np.float32)
    
    # Normalize 
    img_array = img_array / 255.0
    
   
    img_tensor = torch.from_numpy(img_array).unsqueeze(0)  # [1, H, W]
    
   
    img_tensor = F.interpolate(
        img_tensor.unsqueeze(0),  # [1, 1, H, W]
        size=(img_size, img_size),
        mode='bilinear',
        align_corners=False
    ).squeeze(0)  # [1, H, W]
    
    
    img_tensor = (img_tensor - 0.5) / 0.5
    
    return img_tensor, img_array  

def load_mask(mask_path, img_size):
    """Load and preprocess a mask"""
    mask = Image.open(mask_path).convert('L')
    mask_array = np.array(mask, dtype=np.uint8)
    
    # Convert to binary (0 or 1)
    mask_binary = (mask_array > 0).astype(np.float32)
    
   
    mask_tensor = torch.from_numpy(mask_binary).unsqueeze(0)  # [1, H, W]
    
    # Resize 
    mask_tensor = F.interpolate(
        mask_tensor.unsqueeze(0),  # [1, 1, H, W]
        size=(img_size, img_size),
        mode='nearest'
    ).squeeze(0)  # [1, H, W]
    
   
    mask_np = mask_tensor.squeeze(0).numpy()  # [H, W]
    
    return mask_np

def predict(model, img_tensor, device, threshold=0.5):
    """Run inference on a single image"""
    model.eval()
    with torch.no_grad():
       
        img_batch = img_tensor.unsqueeze(0).to(device)
        
        # Forward pass
        output = model(img_batch)  
        
        # Apply sigmoid 
        if output.dim() == 4:
            output = output.squeeze(1) 
        
        prob = torch.sigmoid(output)  # [1, H, W]
        
        # Convert to binary 
        pred_binary = (prob > threshold).float()
        
       
        prob_np = prob.squeeze(0).cpu().numpy()  # [H, W]
        pred_np = pred_binary.squeeze(0).cpu().numpy()  # [H, W]
    
    return prob_np, pred_np

def create_overlay(image, mask, prediction, alpha=0.5):
    """Create an overlay visualization of image, ground truth mask, and prediction"""
    # Convert grayscale image to RGB 
    if len(image.shape) == 2:
        img_rgb = np.stack([image, image, image], axis=-1)
    else:
        img_rgb = image.copy()
    
    # Normalize 
    if img_rgb.max() > 1.0:
        img_rgb = img_rgb / 255.0
    
   
    overlay = img_rgb.copy()
    
   
    green = np.array([0, 1, 0])
    red = np.array([1, 0, 0])
    yellow = np.array([1, 1, 0])
    
    # Ground truth mask (green)
    gt_mask = mask > 0.5
    if np.any(gt_mask):
        overlay[gt_mask] = green * alpha + img_rgb[gt_mask] * (1 - alpha)
    
    # Prediction mask (red)
    pred_mask = prediction > 0.5
    if np.any(pred_mask):
        overlay[pred_mask] = red * alpha + overlay[pred_mask] * (1 - alpha)
    
    # Overlap (yellow) 
    overlap_mask = gt_mask & pred_mask
    if np.any(overlap_mask):
        overlay[overlap_mask] = yellow * alpha + img_rgb[overlap_mask] * (1 - alpha)
    
    return overlay

def visualize_predictions(model, device, csv_path, images_dir, masks_dir, 
                          num_images, img_size, output_path):
    """Main visualization function"""
    # Load CSV
    df = pd.read_csv(csv_path)
    print(f"\nLoaded CSV with {len(df)} image-mask pairs")
    
   
    selected_df = df.sample(n=min(num_images, len(df)), random_state=20).reset_index(drop=True)
    print(f"Selected {len(selected_df)} images for visualization")
    
   
    fig, axes = plt.subplots(num_images, 4, figsize=(16, 4 * num_images))
    
   
    if num_images == 1:
        axes = axes.reshape(1, -1)
    
    
    col_titles = ['Original Image', 'Ground Truth Mask', 'Prediction', 'Overlay']
    for col_idx, title in enumerate(col_titles):
        axes[0, col_idx].set_title(title, fontsize=14, fontweight='bold')
    
    # Process each image
    for row_idx, (_, row) in enumerate(selected_df.iterrows()):
        img_name = row['ImageId']
        mask_name = row['MaskId']
        
        img_path = os.path.join(images_dir, img_name)
        mask_path = os.path.join(masks_dir, mask_name)
        
        print(f"\nProcessing {row_idx + 1}/{num_images}: {img_name}")
        
        # Load and preprocess
        img_tensor, img_original = load_and_preprocess_image(img_path, img_size)
        mask_np = load_mask(mask_path, img_size)
        
        # Run prediction
        prob_np, pred_np = predict(model, img_tensor, device, threshold=THRESHOLD)
        
        # Resize
        img_for_viz = Image.fromarray((img_original * 255).astype(np.uint8))
        img_for_viz = img_for_viz.resize((img_size, img_size), Image.Resampling.LANCZOS)
        img_for_viz = np.array(img_for_viz) / 255.0
        
        # Create overlay
        overlay = create_overlay(img_for_viz, mask_np, pred_np, alpha=0.4)
        
        # Plot 
        axes[row_idx, 0].imshow(img_for_viz, cmap='gray')
        axes[row_idx, 0].axis('off')
        axes[row_idx, 0].set_title(f'Image: {img_name[:30]}...', fontsize=10)
        
        # Plot
        axes[row_idx, 1].imshow(mask_np, cmap='gray', vmin=0, vmax=1)
        axes[row_idx, 1].axis('off')
        
        # Plot 
        axes[row_idx, 2].imshow(pred_np, cmap='gray', vmin=0, vmax=1)
        axes[row_idx, 2].axis('off')
        
        # Plot in fourth column: Overlay
        axes[row_idx, 3].imshow(overlay)
        axes[row_idx, 3].axis('off')
    
   
    legend_elements = [
        mpatches.Patch(color='green', alpha=0.4, label='Ground Truth'),
        mpatches.Patch(color='red', alpha=0.4, label='Prediction'),
        mpatches.Patch(color='yellow', alpha=0.4, label='Overlap')
    ]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
    
   
    plt.tight_layout()
    
    # Save as PNG
    print(f"\nSaving visualization to: {output_path}")
    plt.savefig(output_path, format='png', dpi=300, bbox_inches='tight')
    print("Visualization saved successfully!")
    
    plt.close()

def main():
    """Main function"""
    print("=" * 60)
    print("Segmentation Visualization Script")
    print("=" * 60)
    
   
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"ERROR: Checkpoint not found at: {CHECKPOINT_PATH}")
        print("Please update CHECKPOINT_PATH in the script.")
        return
    
    # Load model
    model, device, config = load_model(CHECKPOINT_PATH, CONFIG_PATH)
    
   
    img_size = config.get('training', {}).get('img_size', IMG_SIZE)
    print(f"Using image size: {img_size}")
    
    # Run visualization
    visualize_predictions(
        model=model,
        device=device,
        csv_path=CSV_PATH,
        images_dir=IMAGES_DIR,
        masks_dir=MASKS_DIR,
        num_images=NUM_IMAGES,
        img_size=img_size,
        output_path=OUTPUT_PNG_PATH
    )
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)

if __name__ == '__main__':
    main()

