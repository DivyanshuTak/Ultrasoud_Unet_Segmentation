import os
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from dataset import SegmentationDataset
from model import UNet

#  wandb only if true 
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("wandb not available. Install with: pip install wandb")


def dice_loss(pred, target, smooth=1e-6):
    """Dice loss for segmentation (expects logits)"""
    pred = torch.sigmoid(pred)  # Apply sigmoid to logits
    # Handle shape differences: 
    if pred.dim() == 4 and target.dim() == 3:
        pred = pred.squeeze(1)  # [B, 1, H, W] -> [B, H, W]
    elif pred.dim() == 4 and target.dim() == 4:
        pred = pred.squeeze(1)  # [B, 1, H, W] -> [B, H, W]
        target = target.squeeze(1) if target.size(1) == 1 else target
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    dice = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
    return 1 - dice


def calculate_metrics(pred, target, threshold=0.5):
    """Calculate segmentation metrics (expects logits)"""
    pred = torch.sigmoid(pred)  # Apply sigmoid to logits to get probabilities
    # Handle shape differences: pred might be [B, 1, H, W], target might be [B, H, W]
    if pred.dim() == 4 and target.dim() == 3:
        pred = pred.squeeze(1)  # [B, 1, H, W] -> [B, H, W]
    elif pred.dim() == 4 and target.dim() == 4:
        pred = pred.squeeze(1)  # [B, 1, H, W] -> [B, H, W]
        target = target.squeeze(1) if target.size(1) == 1 else target
    pred_binary = (pred > threshold).float()
    target_binary = target.float()
    
    # Flatten
    pred_flat = pred_binary.view(-1)
    target_flat = target_binary.view(-1)
    
    # Dice score
    intersection = (pred_flat * target_flat).sum()
    dice = (2. * intersection) / (pred_flat.sum() + target_flat.sum() + 1e-6)
    
    # IoU
    union = pred_flat.sum() + target_flat.sum() - intersection
    iou = intersection / (union + 1e-6)
    
    # Precision, Recall
    tp = intersection
    fp = pred_flat.sum() - intersection
    fn = target_flat.sum() - intersection
    
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    
    return {
        'dice': dice.item(),
        'iou': iou.item(),
        'precision': precision.item(),
        'recall': recall.item()
    }


def train_epoch(model, dataloader, criterion, optimizer, device, loss_config, use_wandb=False):
    model.train()
    total_loss = 0
    total_bce = 0
    total_dice = 0
    
    all_metrics = {'dice': [], 'iou': [], 'precision': [], 'recall': []}
    
    for images, masks in tqdm(dataloader, desc="Training"):
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        
        # Calculate individual losses
        losses = {}
        total_batch_loss = 0
        
        if loss_config.get('use_bce', True):
            # BCEWithLogitsLoss can handle [B, 1, H, W] pred and [B, H, W] target (broadcasts)
            # But to be safe, ensure shapes match
            if outputs.dim() == 4 and masks.dim() == 3:
                masks_expanded = masks.unsqueeze(1)  # [B, H, W] -> [B, 1, H, W]
                bce = criterion(outputs, masks_expanded)
            else:
                bce = criterion(outputs, masks)
            losses['bce'] = bce
            total_batch_loss += loss_config.get('bce_weight', 1.0) * bce
        
        if loss_config.get('use_dice', True):
            dice = dice_loss(outputs, masks)
            losses['dice'] = dice
            total_batch_loss += loss_config.get('dice_weight', 1.0) * dice
        
        total_batch_loss.backward()
        optimizer.step()
        
        # Accumulate losses
        total_loss += total_batch_loss.item()
        if 'bce' in losses:
            total_bce += losses['bce'].item()
        if 'dice' in losses:
            total_dice += losses['dice'].item()
        
        # Calculate metrics
        metrics = calculate_metrics(outputs, masks)
        for key in all_metrics:
            all_metrics[key].append(metrics[key])
    
    # Average metrics
    avg_metrics = {k: np.mean(v) for k, v in all_metrics.items()}
    
    return {
        'loss': total_loss / len(dataloader),
        'bce': total_bce / len(dataloader) if loss_config.get('use_bce', True) else 0,
        'dice_loss': total_dice / len(dataloader) if loss_config.get('use_dice', True) else 0,
        **avg_metrics
    }


def validate(model, dataloader, criterion, device, loss_config):
    model.eval()
    total_loss = 0
    total_bce = 0
    total_dice = 0
    
    all_metrics = {'dice': [], 'iou': [], 'precision': [], 'recall': []}
    
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Validation"):
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            
            # Calculate losses
            losses = {}
            total_batch_loss = 0
            
            if loss_config.get('use_bce', True):
                # BCEWithLogitsLoss can handle [B, 1, H, W] pred and [B, H, W] target (broadcasts)
                # But to be safe, ensure shapes match
                if outputs.dim() == 4 and masks.dim() == 3:
                    masks_expanded = masks.unsqueeze(1)  # [B, H, W] -> [B, 1, H, W]
                    bce = criterion(outputs, masks_expanded)
                else:
                    bce = criterion(outputs, masks)
                losses['bce'] = bce
                total_batch_loss += loss_config.get('bce_weight', 1.0) * bce
            
            if loss_config.get('use_dice', True):
                dice = dice_loss(outputs, masks)
                losses['dice'] = dice
                total_batch_loss += loss_config.get('dice_weight', 1.0) * dice
            
            total_loss += total_batch_loss.item()
            if 'bce' in losses:
                total_bce += losses['bce'].item()
            if 'dice' in losses:
                total_dice += losses['dice'].item()
            
            # Calculate metrics
            metrics = calculate_metrics(outputs, masks)
            for key in all_metrics:
                all_metrics[key].append(metrics[key])
    
    # Average metrics
    avg_metrics = {k: np.mean(v) for k, v in all_metrics.items()}
    
    return {
        'loss': total_loss / len(dataloader),
        'bce': total_bce / len(dataloader) if loss_config.get('use_bce', True) else 0,
        'dice_loss': total_dice / len(dataloader) if loss_config.get('use_dice', True) else 0,
        **avg_metrics
    }


def test(model, dataloader, criterion, device, loss_config):
    """Evaluate on test set"""
    return validate(model, dataloader, criterion, device, loss_config)


def main():
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize wandb (will watch model after it's created)
    wandb_config = config.get('wandb', {})
    use_wandb = wandb_config.get('enabled', False) and WANDB_AVAILABLE
    
    if use_wandb:
        wandb.init(
            project=wandb_config.get('project', 'unet-segmentation'),
            name=wandb_config.get('run_name', None),
            config=config,
            tags=wandb_config.get('tags', [])
        )
    
    # Create datasets with train/val/test split
    import pandas as pd
    from sklearn.model_selection import train_test_split
    
    df = pd.read_csv(config['data']['csv_path'])
    
    # First split: train + (val + test)
    train_df, temp_df = train_test_split(
        df, 
        test_size=(1 - config['split']['train']), 
        random_state=config['split']['random_seed']
    )
    
    # Second split: val and test
    val_size = config['split']['val'] / (config['split']['val'] + config['split']['test'])
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - val_size),
        random_state=config['split']['random_seed']
    )
    
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Save test set IDs to CSV for later inference
    save_dir = config['checkpoints']['save_dir']
    os.makedirs(save_dir, exist_ok=True)
    test_csv_path = os.path.join(save_dir, 'test_set_ids.csv')
    test_df.to_csv(test_csv_path, index=False)
    print(f"Saved test set IDs to: {test_csv_path}")
    print(f"Test set CSV contains {len(test_df)} samples")
    
    # Augmentation config
    aug_config = config.get('augmentation', {}) if config.get('augmentation', {}).get('enabled', False) else None
    
    img_size = int(config['training']['img_size'])
    
    train_dataset = SegmentationDataset(
        train_df, 
        config['data']['images_dir'], 
        config['data']['masks_dir'], 
        img_size=img_size, 
        augment=config.get('augmentation', {}).get('enabled', False),
        aug_config=aug_config
    )
    val_dataset = SegmentationDataset(
        val_df, 
        config['data']['images_dir'], 
        config['data']['masks_dir'],
        img_size=img_size, 
        augment=False
    )
    test_dataset = SegmentationDataset(
        test_df,
        config['data']['images_dir'],
        config['data']['masks_dir'],
        img_size=img_size,
        augment=False
    )
    
    batch_size = int(config['training']['batch_size'])
    num_workers = int(config['training']['num_workers'])
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    # Model
    model = UNet(
        n_channels=config['model']['n_channels'], 
        n_classes=config['model']['n_classes'],
        encoder_name=config['model'].get('encoder_name', 'resnet34'),
        encoder_weights=config['model'].get('encoder_weights', 'imagenet')
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Watch model in wandb
    if use_wandb:
        wandb.watch(model, log='gradients', log_freq=100)
    
    # Loss configuration
    loss_config = config.get('loss', {
        'use_bce': True,
        'use_dice': True,
        'bce_weight': 1.0,
        'dice_weight': 1.0
    })
    
    # Loss function - use BCEWithLogitsLoss for numerical stability
    criterion = nn.BCEWithLogitsLoss()
    
    optimizer_config = config.get('optimizer', {})
    # Ensure learning rate is a float
    learning_rate = float(config['training']['learning_rate'])
    
    if optimizer_config.get('type', 'adam').lower() == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=learning_rate,
            weight_decay=float(optimizer_config.get('weight_decay', 0))
        )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            weight_decay=float(optimizer_config.get('weight_decay', 0)),
            momentum=0.9
        )
    
    scheduler_config = config.get('scheduler', {})
    if scheduler_config.get('type') == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode=scheduler_config.get('mode', 'min'),
            factor=scheduler_config.get('factor', 0.5), 
            patience=scheduler_config.get('patience', 5)
        )
    else:
        scheduler = None
    
    start_epoch = 0
    best_val_loss = float('inf')
    best_val_dice = 0.0
    
    resume_path = config['checkpoints'].get('resume')
    if resume_path and os.path.exists(resume_path):
        checkpoint = torch.load(resume_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_val_loss = checkpoint.get('val_loss', float('inf'))
        best_val_dice = checkpoint.get('val_dice', 0.0)
        print(f"Resumed from epoch {start_epoch}")
    
    save_dir = config['checkpoints']['save_dir']
    os.makedirs(save_dir, exist_ok=True)
    
    # Training loop
    num_epochs = int(config['training']['epochs'])
    for epoch in range(start_epoch, num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, loss_config, use_wandb)
        val_metrics = validate(model, val_loader, criterion, device, loss_config)
        
        if scheduler:
            scheduler.step(val_metrics['loss'])
        
        # Print metrics
        print(f"Train - Loss: {train_metrics['loss']:.4f}, Dice: {train_metrics['dice']:.4f}, IoU: {train_metrics['iou']:.4f}")
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, Dice: {val_metrics['dice']:.4f}, IoU: {val_metrics['iou']:.4f}")
        print(f"Val   - Precision: {val_metrics['precision']:.4f}, Recall: {val_metrics['recall']:.4f}")
        
        # Log to wandb
        if use_wandb:
            log_dict = {
                'epoch': epoch + 1,
                'train/loss': train_metrics['loss'],
                'train/dice': train_metrics['dice'],
                'train/iou': train_metrics['iou'],
                'train/precision': train_metrics['precision'],
                'train/recall': train_metrics['recall'],
                'val/loss': val_metrics['loss'],
                'val/dice': val_metrics['dice'],
                'val/iou': val_metrics['iou'],
                'val/precision': val_metrics['precision'],
                'val/recall': val_metrics['recall'],
                'lr': optimizer.param_groups[0]['lr']
            }
            
            # Add individual loss components
            if loss_config.get('use_bce', True):
                log_dict['train/bce_loss'] = train_metrics['bce']
                log_dict['val/bce_loss'] = val_metrics['bce']
            if loss_config.get('use_dice', True):
                log_dict['train/dice_loss'] = train_metrics['dice_loss']
                log_dict['val/dice_loss'] = val_metrics['dice_loss']
            
            wandb.log(log_dict, step=epoch + 1)
        
        # Save best model (based on validation loss or dice)
        save_best = False
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            save_best = True
        
        if val_metrics['dice'] > best_val_dice:
            best_val_dice = val_metrics['dice']
            if not save_best:  # Only save if loss didn't improve
                save_best = True
        
        if save_best:
            # Extract values for f-string to avoid syntax error
            val_loss = val_metrics['loss']
            val_dice = val_metrics['dice']
            val_iou = val_metrics['iou']
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_dice': val_dice,
                'val_iou': val_iou,
            }, os.path.join(save_dir, f'onlyspleencases_spleen_best_model_{epoch+1}_val_loss_{val_loss:.4f}_val_dice_{val_dice:.4f}_val_iou_{val_iou:.4f}.pt'))
            print(f"Saved best model (val_loss: {val_metrics['loss']:.4f}, val_dice: {val_metrics['dice']:.4f})")
            
            if use_wandb:
                wandb.summary['best_val_loss'] = val_metrics['loss']
                wandb.summary['best_val_dice'] = val_metrics['dice']
                wandb.summary['best_val_iou'] = val_metrics['iou']
        
        # Save checkpoint
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_metrics['loss'],
            'val_dice': val_metrics['dice'],
            'val_iou': val_metrics['iou'],
        }, os.path.join(save_dir, 'last_checkpoint.pt'))
    
    # Final test evaluation
    print("\n" + "="*50)
    print("Evaluating on test set...")
    test_metrics = test(model, test_loader, criterion, device, loss_config)
    print(f"Test - Loss: {test_metrics['loss']:.4f}, Dice: {test_metrics['dice']:.4f}, IoU: {test_metrics['iou']:.4f}")
    print(f"Test - Precision: {test_metrics['precision']:.4f}, Recall: {test_metrics['recall']:.4f}")
    print("="*50)
    
    if use_wandb:
        wandb.log({
            'test/loss': test_metrics['loss'],
            'test/dice': test_metrics['dice'],
            'test/iou': test_metrics['iou'],
            'test/precision': test_metrics['precision'],
            'test/recall': test_metrics['recall'],
        })
        wandb.summary['test_loss'] = test_metrics['loss']
        wandb.summary['test_dice'] = test_metrics['dice']
        wandb.summary['test_iou'] = test_metrics['iou']
        wandb.finish()


if __name__ == '__main__':
    main()
