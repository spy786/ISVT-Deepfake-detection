import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import os

from config import config
from model import ISTVT
from dataset import create_dataloaders
from utils import setup_logging, save_checkpoint, compute_metrics

def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    
    for batch_idx, (videos, labels) in enumerate(pbar):
        videos = videos.to(device)  # (B, T, C, H, W)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        try:
            outputs = model(videos)
            loss = criterion(outputs, labels.float().unsqueeze(1))   # BCE expects float & same shape
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
            # Get predictions for BCE
            preds = (torch.sigmoid(outputs.squeeze(1)) > 0.5).long()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{total_loss/(batch_idx+1):.4f}'
            })
            
        except RuntimeError as e:
            print(f"Training error: {e}")
            continue
    
    avg_loss = total_loss / len(dataloader)
    metrics = compute_metrics(all_labels, all_preds)
    
    return avg_loss, metrics

def validate(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for videos, labels in tqdm(dataloader, desc='Validation'):
            videos = videos.to(device)
            labels = labels.to(device)
            
            try:
                outputs = model(videos)
                loss = criterion(outputs, labels.float().unsqueeze(1))
                
                total_loss += loss.item()
                
                # Get predictions and probabilities for BCE
                probs = torch.sigmoid(outputs.squeeze(1))
                preds = (probs > 0.5).long()

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())  # Probability of fake class
                
            except RuntimeError as e:
                print(f"Validation error: {e}")
                continue
    
    avg_loss = total_loss / len(dataloader)
    metrics = compute_metrics(all_labels, all_preds, all_probs)
    
    return avg_loss, metrics

def main():
    """Main training function"""
    logger = setup_logging()
    logger.info("Starting ISTVT training")
    
    # Set device
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create dataloaders
    dataloaders = create_dataloaders(config)
    logger.info(f"Created dataloaders: {[len(dl) for dl in dataloaders.values()]} batches")
    
    # Create model
    model = ISTVT(config).to(device)
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=config.learning_rate,
        momentum=0.9,
        weight_decay=config.weight_decay
    )
    
    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs)
    
    # Tensorboard writer
    writer = SummaryWriter(log_dir=config.log_dir)
    
    # Training loop
    best_acc = 0.0
    
    for epoch in range(config.num_epochs):
        logger.info(f"\nEpoch {epoch+1}/{config.num_epochs}")
        
        # Warmup learning rate
        if epoch < config.warmup_epochs:
            warmup_factor = (epoch + 1) / config.warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = config.learning_rate * warmup_factor
        
        # Train
        train_loss, train_metrics = train_epoch(
            model, dataloaders['train'], criterion, optimizer, device, epoch+1
        )
        
        # Validate
        val_loss, val_metrics = validate(
            model, dataloaders['val'], criterion, device
        )
        
        # Update scheduler
        if epoch >= config.warmup_epochs:
            scheduler.step()
        
        # Log metrics
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Val', val_loss, epoch)
        writer.add_scalar('Accuracy/Train', train_metrics['accuracy'], epoch)
        writer.add_scalar('Accuracy/Val', val_metrics['accuracy'], epoch)
        writer.add_scalar('AUC/Val', val_metrics['auc'], epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
        # Print metrics
        logger.info(f"Train Loss: {train_loss:.4f}, Acc: {train_metrics['accuracy']:.4f}")
        logger.info(f"Val Loss: {val_loss:.4f}, Acc: {val_metrics['accuracy']:.4f}, "
                   f"AUC: {val_metrics['auc']:.4f}, F1: {val_metrics['f1']:.4f}")
        
        # Save best model
        if val_metrics['accuracy'] > best_acc:
            best_acc = val_metrics['accuracy']
            save_checkpoint(
                model, optimizer, epoch, best_acc,
                os.path.join(config.checkpoint_dir, 'best_model.pth')
            )
            logger.info(f"New best model saved! Accuracy: {best_acc:.4f}")
        
        # Save regular checkpoint
        if (epoch + 1) % 10 == 0:
            save_checkpoint(
                model, optimizer, epoch, best_acc,
                os.path.join(config.checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
            )
    
    logger.info(f"Training completed! Best accuracy: {best_acc:.4f}")
    writer.close()

if __name__ == '__main__':
    main()