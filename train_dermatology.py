"""
MedSAM2 Dermatology Training Script

Training MedSAM2 model for skin lesion segmentation following the official MedSAM2 training approach.

Author: Capstone Team
Date: October 25, 2025
Framework: MedSAM2
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
from pathlib import Path
import logging
import wandb
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
import yaml

# Import MedSAM2 components
from sam2.build_sam import build_sam2
from sam2.sam2_video_trainer import SAM2VideoTrainer

# Import Hydra for config management
from hydra import initialize_config_module
from hydra.core.global_hydra import GlobalHydra


class EarlyStopping:
    """Early stopping to stop training when validation metric stops improving"""
    
    def __init__(self, patience: int = 7, min_delta: float = 0.0, mode: str = 'max', verbose: bool = True):
        """
        Args:
            patience: Number of epochs to wait after last improvement
            min_delta: Minimum change to qualify as an improvement
            mode: 'max' for metrics like dice/iou, 'min' for metrics like loss
            verbose: Whether to print early stopping messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
        
        if mode not in ['min', 'max']:
            raise ValueError(f"mode must be 'min' or 'max', got {mode}")
    
    def __call__(self, current_score: float, epoch: int) -> bool:
        """
        Check if early stopping criteria is met
        
        Args:
            current_score: Current validation metric value
            epoch: Current epoch number
            
        Returns:
            True if training should stop, False otherwise
        """
        if self.best_score is None:
            # First epoch
            self.best_score = current_score
            self.best_epoch = epoch
            if self.verbose:
                logging.info(f"🎯 Early stopping baseline set: {current_score:.6f}")
            return False
        
        # Check if there's improvement
        if self.mode == 'max':
            improved = current_score > (self.best_score + self.min_delta)
        else:  # mode == 'min'
            improved = current_score < (self.best_score - self.min_delta)
        
        if improved:
            # Model improved
            self.best_score = current_score
            self.best_epoch = epoch
            self.counter = 0
            if self.verbose:
                logging.info(f"✅ Validation improved to {current_score:.6f}")
        else:
            # No improvement
            self.counter += 1
            if self.verbose:
                logging.info(f"⏳ No improvement for {self.counter}/{self.patience} epochs (best: {self.best_score:.6f} at epoch {self.best_epoch})")
            
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    logging.info(f"🛑 Early stopping triggered! No improvement for {self.patience} epochs")
                    logging.info(f"Best score: {self.best_score:.6f} at epoch {self.best_epoch}")
                return True
        
        return False
    
    def reset(self):
        """Reset early stopping state"""
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0


class DermatologyDataset(Dataset):
    """Dataset for dermatology images compatible with MedSAM2 training
    
    Supports structure with class folders:
        images/
            Acne/
                image1.jpg
            Eczema/
                image2.jpg
        masks/
            Acne/
                image1.png
            Eczema/
                image2.png
    """
    
    def __init__(self, images_dir: str, masks_dir: str, image_size: int = 512):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.image_size = image_size
        
        self.samples = []
        
        # Check if we have class folders or direct images
        has_class_folders = any(d.is_dir() for d in self.images_dir.iterdir() if not d.name.startswith('.'))
        
        if has_class_folders:
            # Load from class folders structure
            class_folders = [d for d in self.images_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
            
            for class_folder in class_folders:
                class_name = class_folder.name
                
                # Get all images in this class folder
                image_files = list(class_folder.glob('*.jpg')) + list(class_folder.glob('*.png')) + list(class_folder.glob('*.jpeg'))
                
                for img_file in image_files:
                    # Look for corresponding mask in masks/class_name/
                    mask_file = self.masks_dir / class_name / f"{img_file.stem}.png"
                    
                    if not mask_file.exists():
                        # Try .jpg extension for mask
                        mask_file = self.masks_dir / class_name / f"{img_file.stem}.jpg"
                    
                    if mask_file.exists():
                        self.samples.append({
                            'image_path': img_file,
                            'mask_path': mask_file,
                            'class': class_name
                        })
            
            logging.info(f"Dataset loaded: {len(self.samples)} samples from {len(class_folders)} classes")
        else:
            # Load from flat directory structure (backward compatibility)
            image_files = list(self.images_dir.glob('*.png')) + list(self.images_dir.glob('*.jpg'))
            
            for img_file in image_files:
                mask_file = self.masks_dir / img_file.name
                
                if mask_file.exists():
                    self.samples.append({
                        'image_path': img_file,
                        'mask_path': mask_file,
                        'class': 'unknown'
                    })
                else:
                    # Try alternative naming
                    mask_file_alt = self.masks_dir / f"{img_file.stem}_mask.png"
                    if mask_file_alt.exists():
                        self.samples.append({
                            'image_path': img_file,
                            'mask_path': mask_file_alt,
                            'class': 'unknown'
                        })
            
            logging.info(f"Dataset loaded: {len(self.samples)} samples from flat directory")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = sample['image_path']
        mask_path = sample['mask_path']
        
        # Load and resize
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        mask = mask.resize((self.image_size, self.image_size), Image.NEAREST)
        
        # Convert to arrays
        image = np.array(image)
        mask = np.array(mask)
        mask = (mask > 128).astype(np.float32)
        
        return {
            'image': image,
            'mask': mask,
            'filename': img_path.name,
            'class': sample['class']
        }


class MedSAM2DermatologyTrainer:
    """Trainer for MedSAM2 on dermatology dataset"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device(config['device'])
        
        # Setup logging
        self.setup_logging()
        
        # Initialize model
        self.model = self.build_model()
        
        # Setup datasets and dataloaders
        self.train_loader, self.val_loader = self.setup_data()
        
        # Setup optimizer and scheduler
        self.optimizer, self.scheduler = self.setup_optimizer()
        
        # Setup loss functions
        self.setup_loss_functions()
        
        # Initialize training state
        self.best_val_dice = 0.0
        self.history = {
            'train_loss': [], 'train_dice': [], 'train_iou': [],
            'val_loss': [], 'val_dice': [], 'val_iou': []
        }
        
        # Setup early stopping
        self.early_stopping = None
        if config.get('early_stopping_patience', 0) > 0:
            self.early_stopping = EarlyStopping(
                patience=config['early_stopping_patience'],
                min_delta=config.get('early_stopping_min_delta', 0.001),
                mode=config.get('early_stopping_mode', 'max'),  # 'max' for dice/iou, 'min' for loss
                verbose=config.get('early_stopping_verbose', True)
            )
            logging.info(f"✅ Early stopping enabled with patience={config['early_stopping_patience']}")
        
        # Setup WandB
        if config.get('use_wandb', False):
            self.setup_wandb()
    
    def setup_logging(self):
        """Setup logging configuration"""
        log_dir = Path(self.config.get('log_dir', 'logs'))
        log_dir.mkdir(exist_ok=True, parents=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f'training_{timestamp}.log'
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        logging.info("=" * 60)
        logging.info("MedSAM2 Dermatology Training")
        logging.info("=" * 60)
        for key, value in self.config.items():
            logging.info(f"{key:20s}: {value}")
        logging.info("=" * 60)
    
    def build_model(self):
        """Build MedSAM2 model following official approach"""
        logging.info("Building MedSAM2 model...")
        
        # Reset Hydra if it's already initialized
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()
        
        # Initialize Hydra with config directory
        from hydra import initialize_config_dir
        import os
        config_dir = os.path.join(os.path.dirname(__file__), 'sam2', 'configs')
        initialize_config_dir(config_dir=config_dir, version_base="1.2")
        
        # Build SAM2 model - use just the config filename without path
        model = build_sam2(
            config_file=self.config['sam2_config'],
            ckpt_path=self.config['sam2_checkpoint'],
            device=self.device,
            mode="train"  # Use train mode
        )
        
        # Freeze image encoder (recommended by MedSAM2)
        for name, param in model.image_encoder.named_parameters():
            param.requires_grad = False
        
        # Train mask decoder and prompt encoder
        for name, param in model.sam_mask_decoder.named_parameters():
            param.requires_grad = True
        
        for name, param in model.sam_prompt_encoder.named_parameters():
            param.requires_grad = True
        
        model.to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logging.info(f"✅ MedSAM2 loaded successfully")
        logging.info(f"Total parameters: {total_params:,}")
        logging.info(f"Trainable parameters: {trainable_params:,}")
        logging.info(f"Frozen parameters: {total_params - trainable_params:,}")
        
        return model
    
    def setup_data(self):
        """Setup datasets and dataloaders"""
        logging.info("Setting up datasets...")
        
        dataset_path = Path(self.config['dataset_path'])
        
        train_dataset = DermatologyDataset(
            images_dir=dataset_path / 'train' / 'images',
            masks_dir=dataset_path / 'train' / 'masks',
            image_size=self.config['image_size']
        )
        
        val_dataset = DermatologyDataset(
            images_dir=dataset_path / 'val' / 'images',
            masks_dir=dataset_path / 'val' / 'masks',
            image_size=self.config['image_size']
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config.get('num_workers', 4),
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config.get('num_workers', 4),
            pin_memory=True
        )
        
        logging.info(f"Train samples: {len(train_dataset)}, Batches: {len(train_loader)}")
        logging.info(f"Val samples: {len(val_dataset)}, Batches: {len(val_loader)}")
        
        return train_loader, val_loader
    
    def setup_optimizer(self):
        """Setup optimizer and learning rate scheduler"""
        # Only optimize trainable parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config['num_epochs']
        )
        
        logging.info(f"✅ Optimizer setup: AdamW (lr={self.config['learning_rate']})")
        
        return optimizer, scheduler
    
    def setup_loss_functions(self):
        """Setup loss functions"""
        self.bce_loss_fn = nn.BCEWithLogitsLoss()
        self.scaler = torch.cuda.amp.GradScaler(
            enabled=self.config.get('mixed_precision', True)
        )
    
    def dice_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute Dice loss"""
        pred = torch.sigmoid(pred)
        smooth = 1e-6
        
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        dice = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
        
        return 1.0 - dice
    
    def iou_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute IOU (Intersection over Union) loss"""
        pred = torch.sigmoid(pred)
        smooth = 1e-6
        
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum() - intersection
        
        iou = (intersection + smooth) / (union + smooth)
        
        return 1.0 - iou
    
    def compute_dice_score(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Compute Dice score for evaluation"""
        pred = torch.sigmoid(pred)
        pred = (pred > 0.5).float()
        
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        dice = (2. * intersection + 1e-6) / (pred_flat.sum() + target_flat.sum() + 1e-6)
        
        return dice.item()
    
    def compute_iou_score(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Compute IOU score for evaluation"""
        pred = torch.sigmoid(pred)
        pred = (pred > 0.5).float()
        
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum() - intersection
        
        iou = (intersection + 1e-6) / (union + 1e-6)
        
        return iou.item()
    
    def forward_pass(self, images: torch.Tensor, masks: torch.Tensor):
        """Forward pass through MedSAM2"""
        B, C, H, W = images.shape
        
        # Process through image encoder (frozen)
        with torch.no_grad():
            # Resize to 1024x1024 if needed (SAM2 requirement)
            if H != 1024 or W != 1024:
                x = F.interpolate(images, size=(1024, 1024), mode='bilinear', align_corners=False)
            else:
                x = images
            
            # Image encoder returns a dict with vision_features
            encoder_output = self.model.image_encoder(x)
            image_embeddings = encoder_output["vision_features"]
            backbone_fpn = encoder_output.get("backbone_fpn", None)
        
        # Use the prompt encoder to get embeddings (no prompts for automatic segmentation)
        # The prompt encoder needs to know the batch size
        sparse_embeddings, dense_embeddings = self.model.sam_prompt_encoder(
            points=None,
            boxes=None,
            masks=None,
        )
        
        # Expand embeddings to batch size if needed
        if sparse_embeddings.shape[0] != B:
            sparse_embeddings = sparse_embeddings.expand(B, -1, -1)
        if dense_embeddings.shape[0] != B:
            dense_embeddings = dense_embeddings.expand(B, -1, -1, -1)
        
        # Resize dense_embeddings to match image_embeddings spatial size
        _, _, embed_h, embed_w = image_embeddings.shape
        if dense_embeddings.shape[-2:] != (embed_h, embed_w):
            dense_embeddings = F.interpolate(
                dense_embeddings,
                size=(embed_h, embed_w),
                mode='bilinear',
                align_corners=False
            )
        
        # Get position encodings - this gives the correct size for mask decoder
        image_pe = self.model.sam_prompt_encoder.get_dense_pe()
        
        # Resize image_pe to match image_embeddings spatial size if needed
        if image_pe.shape[-2:] != (embed_h, embed_w):
            image_pe = F.interpolate(
                image_pe,
                size=(embed_h, embed_w),
                mode='bilinear',
                align_corners=False
            )
        
        # Prepare high_res_features - mask decoder expects exactly 2 features [feat_s0, feat_s1]
        # Apply the mask decoder's conv layers to project features to the right dimension
        if backbone_fpn is not None and len(backbone_fpn) >= 2:
            # Apply conv_s0 and conv_s1 to project features to correct dimensions
            feat_s0 = self.model.sam_mask_decoder.conv_s0(backbone_fpn[0])
            feat_s1 = self.model.sam_mask_decoder.conv_s1(backbone_fpn[1])
            high_res_features = [feat_s0, feat_s1]
        else:
            high_res_features = None
        
        # Decode masks - returns 4 values: masks, iou_pred, sam_tokens_out, object_score_logits
        low_res_masks, iou_predictions, sam_tokens_out, object_score_logits = self.model.sam_mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
            repeat_image=False,
            high_res_features=high_res_features,
        )
        
        # Upscale to original size
        outputs = F.interpolate(
            low_res_masks,
            size=(H, W),
            mode='bilinear',
            align_corners=False,
        )
        
        return outputs
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        running_loss = 0.0
        running_dice = 0.0
        running_iou = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch} [Train]')
        
        for batch in pbar:
            images = batch['image']
            masks = batch['mask']
            
            # Convert to tensors - handle numpy arrays from DataLoader
            if isinstance(images, list):
                images = torch.stack([torch.from_numpy(img.transpose(2, 0, 1)) for img in images])
            elif isinstance(images, np.ndarray):
                images = torch.from_numpy(images).permute(0, 3, 1, 2)
            else:  # Already a tensor
                if images.dim() == 4 and images.shape[-1] == 3:  # NHWC format
                    images = images.permute(0, 3, 1, 2)
            
            images = images.float() / 255.0
            images = images.to(self.device)
            
            if isinstance(masks, list):
                masks = torch.stack([torch.from_numpy(mask) for mask in masks])
            elif isinstance(masks, np.ndarray):
                masks = torch.from_numpy(masks)
            # else: already a tensor
            
            masks = masks.unsqueeze(1).float().to(self.device)
            
            # Forward pass with mixed precision
            with torch.cuda.amp.autocast(enabled=self.config.get('mixed_precision', True)):
                outputs = self.forward_pass(images, masks)
                
                bce = self.bce_loss_fn(outputs, masks)
                dice = self.dice_loss(outputs, masks)
                iou = self.iou_loss(outputs, masks)
                
                # Combined loss: BCE + Dice + IOU
                total_loss = bce + dice + iou
            
            # Backward pass
            self.optimizer.zero_grad()
            self.scaler.scale(total_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Metrics
            with torch.no_grad():
                dice_score = self.compute_dice_score(outputs, masks)
                iou_score = self.compute_iou_score(outputs, masks)
            
            running_loss += total_loss.item()
            running_dice += dice_score
            running_iou += iou_score
            num_batches += 1
            
            pbar.set_postfix({
                'loss': f"{running_loss/num_batches:.4f}",
                'dice': f"{running_dice/num_batches:.4f}",
                'iou': f"{running_iou/num_batches:.4f}"
            })
        
        return {
            'loss': running_loss / num_batches if num_batches > 0 else 0,
            'dice': running_dice / num_batches if num_batches > 0 else 0,
            'iou': running_iou / num_batches if num_batches > 0 else 0
        }
    
    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()
        
        running_loss = 0.0
        running_dice = 0.0
        running_iou = 0.0
        num_batches = 0
        
        pbar = tqdm(self.val_loader, desc=f'Epoch {epoch} [Val]')
        
        with torch.no_grad():
            for batch in pbar:
                images = batch['image']
                masks = batch['mask']
                
                # Convert to tensors - handle numpy arrays from DataLoader
                if isinstance(images, list):
                    images = torch.stack([torch.from_numpy(img.transpose(2, 0, 1)) for img in images])
                elif isinstance(images, np.ndarray):
                    images = torch.from_numpy(images).permute(0, 3, 1, 2)
                else:  # Already a tensor
                    if images.dim() == 4 and images.shape[-1] == 3:  # NHWC format
                        images = images.permute(0, 3, 1, 2)
                
                images = images.float() / 255.0
                images = images.to(self.device)
                
                if isinstance(masks, list):
                    masks = torch.stack([torch.from_numpy(mask) for mask in masks])
                elif isinstance(masks, np.ndarray):
                    masks = torch.from_numpy(masks)
                # else: already a tensor
                
                masks = masks.unsqueeze(1).float().to(self.device)
                
                with torch.cuda.amp.autocast(enabled=self.config.get('mixed_precision', True)):
                    outputs = self.forward_pass(images, masks)
                    
                    bce = self.bce_loss_fn(outputs, masks)
                    dice = self.dice_loss(outputs, masks)
                    iou = self.iou_loss(outputs, masks)
                    
                    # Combined loss: BCE + Dice + IOU
                    total_loss = bce + dice + iou
                
                dice_score = self.compute_dice_score(outputs, masks)
                iou_score = self.compute_iou_score(outputs, masks)
                
                running_loss += total_loss.item()
                running_dice += dice_score
                running_iou += iou_score
                num_batches += 1
                
                pbar.set_postfix({
                    'loss': f"{running_loss/num_batches:.4f}",
                    'dice': f"{running_dice/num_batches:.4f}",
                    'iou': f"{running_iou/num_batches:.4f}"
                })
        
        return {
            'loss': running_loss / num_batches if num_batches > 0 else 0,
            'dice': running_dice / num_batches if num_batches > 0 else 0,
            'iou': running_iou / num_batches if num_batches > 0 else 0
        }
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint_dir = Path(self.config.get('checkpoint_dir', 'checkpoints'))
        checkpoint_dir.mkdir(exist_ok=True, parents=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_dice': self.best_val_dice,
            'config': self.config
        }
        
        if is_best:
            save_path = checkpoint_dir / 'medsam2_dermatology_best.pth'
            torch.save(checkpoint, save_path)
            logging.info(f"✅ Best model saved: {save_path}")
        
        # Save periodic checkpoint
        if epoch % 5 == 0:
            save_path = checkpoint_dir / f'medsam2_dermatology_epoch_{epoch}.pth'
            torch.save(checkpoint, save_path)
            logging.info(f"💾 Checkpoint saved: {save_path}")
    
    def setup_wandb(self):
        """Setup Weights & Biases logging"""
        # Generate descriptive run name with key hyperparameters
        if 'wandb_name' not in self.config:
            bs = self.config['batch_size']
            epochs = self.config['num_epochs']
            lr = self.config['learning_rate']
            wd = self.config['weight_decay']
            img_size = self.config['image_size']
            
            # Format: medsam2_bs8_20ep_lr1e-4_wd1e-5_img512
            run_name = f"medsam2_bs{bs}_ep{epochs}_lr{lr:.0e}_wd{wd:.0e}_img{img_size}"
        else:
            run_name = self.config['wandb_name']
        
        wandb.init(
            project=self.config.get('wandb_project', 'MedSAM2-Dermatology'),
            entity=self.config.get('wandb_entity', None),
            name=run_name,
            config=self.config,
            tags=[
                f"batch_size_{self.config['batch_size']}",
                f"lr_{self.config['learning_rate']}",
                f"epochs_{self.config['num_epochs']}",
                "with_iou_loss",
                "dermatology",
                "sam2.1_hiera_tiny"
            ]
        )
        logging.info(f"✅ WandB initialized: {run_name}")
    
    def train(self):
        """Main training loop"""
        logging.info("\n🚀 Starting Training")
        logging.info("=" * 60)
        
        for epoch in range(1, self.config['num_epochs'] + 1):
            logging.info(f"\n📊 Epoch {epoch}/{self.config['num_epochs']}")
            logging.info("-" * 50)
            
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate_epoch(epoch)
            
            # Update scheduler
            self.scheduler.step()
            
            # Update history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_dice'].append(train_metrics['dice'])
            self.history['train_iou'].append(train_metrics['iou'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_dice'].append(val_metrics['dice'])
            self.history['val_iou'].append(val_metrics['iou'])
            
            # Log to WandB
            if self.config.get('use_wandb', False):
                wandb_log = {
                    'epoch': epoch,
                    'train/loss': train_metrics['loss'],
                    'train/dice': train_metrics['dice'],
                    'train/iou': train_metrics['iou'],
                    'val/loss': val_metrics['loss'],
                    'val/dice': val_metrics['dice'],
                    'val/iou': val_metrics['iou'],
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                }
                
                # Add early stopping counter if enabled
                if self.early_stopping is not None:
                    wandb_log['early_stopping/counter'] = self.early_stopping.counter
                    wandb_log['early_stopping/best_score'] = self.early_stopping.best_score if self.early_stopping.best_score else 0
                
                wandb.log(wandb_log)
            
            # Print summary
            logging.info(f"\n📈 Results:")
            logging.info(f"Train - Loss: {train_metrics['loss']:.4f}, Dice: {train_metrics['dice']:.4f}, IOU: {train_metrics['iou']:.4f}")
            logging.info(f"Val   - Loss: {val_metrics['loss']:.4f}, Dice: {val_metrics['dice']:.4f}, IOU: {val_metrics['iou']:.4f}")
            
            # Save checkpoints
            is_best = val_metrics['dice'] > self.best_val_dice
            if is_best:
                self.best_val_dice = val_metrics['dice']
                logging.info(f"✅ New best model! Val Dice: {self.best_val_dice:.4f}")
            
            self.save_checkpoint(epoch, is_best=is_best)
            
            # Check early stopping
            if self.early_stopping is not None:
                # Determine which metric to use for early stopping
                early_stop_metric = self.config.get('early_stopping_metric', 'val_dice')
                
                if early_stop_metric == 'val_dice':
                    metric_value = val_metrics['dice']
                elif early_stop_metric == 'val_iou':
                    metric_value = val_metrics['iou']
                elif early_stop_metric == 'val_loss':
                    metric_value = val_metrics['loss']
                else:
                    metric_value = val_metrics['dice']  # Default to dice
                
                # Check if should stop
                should_stop = self.early_stopping(metric_value, epoch)
                
                if should_stop:
                    logging.info(f"\n🛑 Early stopping at epoch {epoch}")
                    logging.info(f"Best {early_stop_metric}: {self.early_stopping.best_score:.6f} at epoch {self.early_stopping.best_epoch}")
                    break
        
        logging.info(f"\n🎉 Training completed!")
        logging.info(f"Best validation Dice: {self.best_val_dice:.4f}")
        
        # Plot and save training history
        self.plot_history()
        
        # WandB finish
        if self.config.get('use_wandb', False):
            wandb.finish()
    
    def plot_history(self):
        """Plot and save training history"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # Loss plot
        axes[0].plot(epochs, self.history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        axes[0].plot(epochs, self.history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Dice plot
        axes[1].plot(epochs, self.history['train_dice'], 'b-', label='Train Dice', linewidth=2)
        axes[1].plot(epochs, self.history['val_dice'], 'r-', label='Val Dice', linewidth=2)
        axes[1].set_title('Dice Score', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Dice Score')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # IOU plot
        axes[2].plot(epochs, self.history['train_iou'], 'b-', label='Train IOU', linewidth=2)
        axes[2].plot(epochs, self.history['val_iou'], 'r-', label='Val IOU', linewidth=2)
        axes[2].set_title('IOU Score', fontsize=14, fontweight='bold')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('IOU Score')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = Path(self.config.get('log_dir', 'logs')) / 'training_history.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"📈 Training history saved: {save_path}")
        
        # WandB logging
        if self.config.get('use_wandb', False):
            wandb.log({"training_history": wandb.Image(str(save_path))})


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train MedSAM2 on Dermatology Dataset')
    
    # Dataset
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Path to the dataset directory')
    parser.add_argument('--image_size', type=int, default=512,
                        help='Image size for training')
    
    # Model
    parser.add_argument('--sam2_config', type=str, 
                        default='configs/sam2.1_hiera_t512.yaml',  # Just filename
                        help='SAM2 config file')
    parser.add_argument('--sam2_checkpoint', type=str,
                        default='checkpoints/sam2.1_hiera_tiny.pt',
                        help='SAM2 checkpoint file')
    
    # Training
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=30,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loader workers')
    parser.add_argument('--mixed_precision', action='store_true',
                        help='Use mixed precision training')
    
    # Logging
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory to save logs')
    
    # Early Stopping
    parser.add_argument('--early_stopping_patience', type=int, default=0,
                        help='Early stopping patience (0 to disable)')
    parser.add_argument('--early_stopping_min_delta', type=float, default=0.001,
                        help='Minimum change to qualify as improvement')
    parser.add_argument('--early_stopping_metric', type=str, default='val_dice',
                        choices=['val_dice', 'val_iou', 'val_loss'],
                        help='Metric to monitor for early stopping')
    parser.add_argument('--early_stopping_mode', type=str, default='max',
                        choices=['max', 'min'],
                        help='Mode for early stopping (max for dice/iou, min for loss)')
    parser.add_argument('--early_stopping_verbose', action='store_true', default=True,
                        help='Print early stopping messages')
    
    # WandB
    parser.add_argument('--use_wandb', action='store_true',
                        help='Use Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='MedSAM2-Dermatology',
                        help='WandB project name')
    parser.add_argument('--wandb_entity', type=str, default=None,
                        help='WandB entity name')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use for training')
    
    return parser.parse_args()


def main():
    """Main function"""
    # Parse arguments
    args = parse_args()
    
    # Create config dict
    config = vars(args)
    
    # Check CUDA availability
    if config['device'] == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA not available, using CPU")
        config['device'] = 'cpu'
    
    # Create trainer and start training
    trainer = MedSAM2DermatologyTrainer(config)
    trainer.train()


if __name__ == '__main__':
    main()
