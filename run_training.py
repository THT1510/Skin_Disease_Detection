"""
Simple script to run MedSAM2 dermatology training
No command line arguments needed - just edit the config below
"""

import torch
from train_dermatology import MedSAM2DermatologyTrainer

# Training Configuration
config = {
    # Dataset path - CHANGE THIS to your dataset location
    'dataset_path': r'C:\Users\hqgho\OneDrive\Máy tính\Final_Project\original_data',
    
    # Model configuration (use just the filename, not full path)
    'sam2_config': 'sam2.1_hiera_t512',  # Just filename without .yaml extension
    'sam2_checkpoint': 'checkpoints/MedSAM2_latest.pt',
    
    # Training parameters - ADJUSTED FOR FULL MODEL TRAINING
    'batch_size': 4,  # Reduced from 8 → 4 (full model needs more memory)
    'num_epochs': 50,  # Max epochs (early stopping will trigger earlier)
    'learning_rate': 5e-5,  # Reduced from 1e-4 → 5e-5 (more stable for full model)
    'weight_decay': 1e-4,
    'image_size': 512,
    
    # Data loading
    'num_workers': 4,  # Reduce to 0 if you have issues with multiprocessing
    
    # Training options
    'mixed_precision': True,  # Use mixed precision for faster training
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    
    # Logging and checkpoints
    'checkpoint_dir': 'checkpoints',
    'log_dir': 'logs',
    
    # Early Stopping Configuration
    # Based on past experiments, model typically peaks around epoch 15-20 
    # and plateaus after ~5-7 epochs without improvement
    'early_stopping_patience': 7,  # Stop if no improvement for 7 epochs
    'early_stopping_min_delta': 0.001,  # 0.1% minimum improvement (0.001 = 0.1%)
    'early_stopping_metric': 'val_dice',  # Monitor validation Dice score
    'early_stopping_mode': 'max',  # Maximize Dice score
    'early_stopping_verbose': True,  # Print progress messages
    
    # WandB - enabled
    'use_wandb': True,
    'wandb_project': 'MedSAM2-Dermatology',
    # 'wandb_name': 'dermatology-segmentation',  # Comment this to auto-generate name with hyperparameters
}

def main():
    print("=" * 60)
    print("MedSAM2 Dermatology Training")
    print("=" * 60)
    
    # Check CUDA
    if torch.cuda.is_available():
        print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print("⚠️  No GPU detected, using CPU (training will be slow)")
    
    print("\nConfiguration:")
    print("-" * 60)
    for key, value in config.items():
        print(f"  {key:20s}: {value}")
    print("=" * 60)
    
    # Create trainer and start training
    print("\n🚀 Initializing trainer...")
    trainer = MedSAM2DermatologyTrainer(config)
    
    print("\n🎯 Starting training loop...")
    trainer.train()
    
    print("\n✅ Training completed!")
    print(f"Best model saved to: checkpoints/medsam2_dermatology_best.pth")
    print(f"Training history saved to: logs/training_history.png")

if __name__ == '__main__':
    main()
