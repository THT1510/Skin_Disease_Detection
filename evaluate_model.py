"""
Evaluate MedSAM2 Model Performance on Test Dataset

This script evaluates the trained model on the test set and provides:
- Overall metrics (Dice, IOU, Precision, Recall)
- Per-class performance
- Confusion analysis
- Performance visualizations
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from pathlib import Path
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from hydra import initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
import json
from datetime import datetime

from sam2.build_sam import build_sam2


class MedSAM2Evaluator:
    """Comprehensive evaluation of MedSAM2 model"""
    
    def __init__(self, model_path, config_name='sam2.1_hiera_t512'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 Using device: {self.device}")
        
        # Build and load model
        self.model = self.build_model(config_name)
        self.load_checkpoint(model_path)
        self.model.eval()
        
        # Metrics storage
        self.results = {
            'per_class': {},
            'overall': {},
            'per_sample': []
        }
    
    def build_model(self, config_name):
        """Build MedSAM2 model"""
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()
        
        config_dir = os.path.join(os.path.dirname(__file__), 'sam2', 'configs')
        initialize_config_dir(config_dir=config_dir, version_base="1.2")
        
        model = build_sam2(
            config_file=config_name,
            ckpt_path='checkpoints/sam2.1_hiera_tiny.pt',
            device=self.device,
            mode="eval"
        )
        
        for param in model.image_encoder.parameters():
            param.requires_grad = False
        
        return model
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        print(f"📦 Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            epoch = checkpoint.get('epoch', 'unknown')
            print(f"✅ Loaded checkpoint from epoch {epoch}")
        else:
            self.model.load_state_dict(checkpoint)
            print("✅ Loaded model state dict")
    
    def preprocess_image(self, image_path, target_size=512):
        """Load and preprocess image"""
        image = Image.open(image_path).convert('RGB')
        image = image.resize((target_size, target_size), Image.BILINEAR)
        image_array = np.array(image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0)
        return image_tensor.to(self.device)
    
    def preprocess_mask(self, mask_path, target_size=512):
        """Load and preprocess mask"""
        mask = Image.open(mask_path).convert('L')
        mask = mask.resize((target_size, target_size), Image.NEAREST)
        mask_array = np.array(mask).astype(np.float32) / 255.0
        mask_tensor = torch.from_numpy(mask_array).unsqueeze(0).unsqueeze(0)
        return mask_tensor.to(self.device)
    
    @torch.no_grad()
    def predict(self, image_tensor):
        """Run inference"""
        B, C, H, W = image_tensor.shape
        
        # Resize to 1024x1024
        if H != 1024 or W != 1024:
            x = F.interpolate(image_tensor, size=(1024, 1024), mode='bilinear', align_corners=False)
        else:
            x = image_tensor
        
        # Image encoder
        encoder_output = self.model.image_encoder(x)
        image_embeddings = encoder_output["vision_features"]
        backbone_fpn = encoder_output.get("backbone_fpn", None)
        
        # Prompt embeddings
        sparse_embeddings, dense_embeddings = self.model.sam_prompt_encoder(
            points=None, boxes=None, masks=None
        )
        
        # Expand to batch size
        if sparse_embeddings.shape[0] != B:
            sparse_embeddings = sparse_embeddings.expand(B, -1, -1)
        if dense_embeddings.shape[0] != B:
            dense_embeddings = dense_embeddings.expand(B, -1, -1, -1)
        
        # Resize embeddings
        _, _, embed_h, embed_w = image_embeddings.shape
        if dense_embeddings.shape[-2:] != (embed_h, embed_w):
            dense_embeddings = F.interpolate(
                dense_embeddings, size=(embed_h, embed_w),
                mode='bilinear', align_corners=False
            )
        
        # Position encodings
        image_pe = self.model.sam_prompt_encoder.get_dense_pe()
        if image_pe.shape[-2:] != (embed_h, embed_w):
            image_pe = F.interpolate(
                image_pe, size=(embed_h, embed_w),
                mode='bilinear', align_corners=False
            )
        
        # High-res features
        if backbone_fpn is not None and len(backbone_fpn) >= 2:
            feat_s0 = self.model.sam_mask_decoder.conv_s0(backbone_fpn[0])
            feat_s1 = self.model.sam_mask_decoder.conv_s1(backbone_fpn[1])
            high_res_features = [feat_s0, feat_s1]
        else:
            high_res_features = None
        
        # Decode masks
        low_res_masks, _, _, _ = self.model.sam_mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
            repeat_image=False,
            high_res_features=high_res_features,
        )
        
        # Upscale and sigmoid
        outputs = F.interpolate(low_res_masks, size=(H, W), mode='bilinear', align_corners=False)
        outputs = torch.sigmoid(outputs)
        
        return outputs
    
    def calculate_metrics(self, pred, target, threshold=0.5):
        """Calculate comprehensive metrics"""
        pred_binary = (pred > threshold).float()
        target_binary = (target > 0.5).float()
        
        pred_flat = pred_binary.cpu().numpy().flatten()
        target_flat = target_binary.cpu().numpy().flatten()
        
        # Basic metrics
        intersection = (pred_binary * target_binary).sum().item()
        pred_sum = pred_binary.sum().item()
        target_sum = target_binary.sum().item()
        union = pred_sum + target_sum - intersection
        
        # Dice score
        dice = (2.0 * intersection + 1e-6) / (pred_sum + target_sum + 1e-6)
        
        # IOU score
        iou = (intersection + 1e-6) / (union + 1e-6)
        
        # Precision, Recall, F1
        precision, recall, f1, _ = precision_recall_fscore_support(
            target_flat, pred_flat, average='binary', zero_division=0
        )
        
        return {
            'dice': dice,
            'iou': iou,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'intersection': intersection,
            'pred_sum': pred_sum,
            'target_sum': target_sum
        }
    
    def evaluate_on_dataset(self, dataset_path, split='test', save_dir='evaluation_results'):
        """Evaluate on entire test dataset"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Get test data
        split_dir = Path(dataset_path) / split
        images_dir = split_dir / 'images'
        masks_dir = split_dir / 'masks'
        
        if not images_dir.exists():
            print(f"❌ Error: {images_dir} does not exist!")
            return
        
        # Collect all images from class folders
        class_folders = [d for d in images_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
        
        print(f"\n{'='*70}")
        print(f"Evaluating on {split.upper()} set")
        print(f"{'='*70}")
        print(f"Found {len(class_folders)} classes")
        
        all_metrics = []
        
        # Process each class
        for class_folder in tqdm(class_folders, desc="Processing classes"):
            class_name = class_folder.name
            
            # Get images for this class
            image_files = list(class_folder.glob('*.jpg')) + \
                         list(class_folder.glob('*.png')) + \
                         list(class_folder.glob('*.jpeg'))
            
            if len(image_files) == 0:
                continue
            
            class_metrics = []
            
            # Process each image
            for image_path in tqdm(image_files, desc=f"  {class_name}", leave=False):
                # Get corresponding mask
                mask_path = masks_dir / class_name / f"{image_path.stem}.png"
                if not mask_path.exists():
                    mask_path = masks_dir / class_name / f"{image_path.stem}.jpg"
                
                if not mask_path.exists():
                    continue
                
                # Predict
                image_tensor = self.preprocess_image(str(image_path))
                mask_tensor = self.preprocess_mask(str(mask_path))
                pred_mask = self.predict(image_tensor)
                
                # Calculate metrics
                metrics = self.calculate_metrics(pred_mask, mask_tensor)
                metrics['class'] = class_name
                metrics['filename'] = image_path.name
                
                class_metrics.append(metrics)
                all_metrics.append(metrics)
            
            # Store per-class results
            if len(class_metrics) > 0:
                self.results['per_class'][class_name] = {
                    'dice': np.mean([m['dice'] for m in class_metrics]),
                    'iou': np.mean([m['iou'] for m in class_metrics]),
                    'precision': np.mean([m['precision'] for m in class_metrics]),
                    'recall': np.mean([m['recall'] for m in class_metrics]),
                    'f1': np.mean([m['f1'] for m in class_metrics]),
                    'count': len(class_metrics)
                }
        
        # Calculate overall metrics
        if len(all_metrics) > 0:
            self.results['overall'] = {
                'dice': np.mean([m['dice'] for m in all_metrics]),
                'iou': np.mean([m['iou'] for m in all_metrics]),
                'precision': np.mean([m['precision'] for m in all_metrics]),
                'recall': np.mean([m['recall'] for m in all_metrics]),
                'f1': np.mean([m['f1'] for m in all_metrics]),
                'total_samples': len(all_metrics)
            }
            
            self.results['overall']['std_dice'] = np.std([m['dice'] for m in all_metrics])
            self.results['overall']['std_iou'] = np.std([m['iou'] for m in all_metrics])
        
        self.results['per_sample'] = all_metrics
        
        # Print results
        self.print_results()
        
        # Save results
        self.save_results(save_dir)
        
        # Create visualizations
        self.create_visualizations(save_dir)
        
        return self.results
    
    def print_results(self):
        """Print evaluation results"""
        print(f"\n{'='*70}")
        print("OVERALL PERFORMANCE")
        print(f"{'='*70}")
        overall = self.results['overall']
        print(f"Total Samples: {overall['total_samples']}")
        print(f"Dice Score:    {overall['dice']:.4f} ± {overall['std_dice']:.4f}")
        print(f"IOU Score:     {overall['iou']:.4f} ± {overall['std_iou']:.4f}")
        print(f"Precision:     {overall['precision']:.4f}")
        print(f"Recall:        {overall['recall']:.4f}")
        print(f"F1 Score:      {overall['f1']:.4f}")
        
        print(f"\n{'='*70}")
        print("PER-CLASS PERFORMANCE")
        print(f"{'='*70}")
        print(f"{'Class':<30} {'Count':>7} {'Dice':>7} {'IOU':>7} {'Prec':>7} {'Recall':>7}")
        print(f"{'-'*70}")
        
        # Sort by Dice score
        sorted_classes = sorted(self.results['per_class'].items(), 
                               key=lambda x: x[1]['dice'], reverse=True)
        
        for class_name, metrics in sorted_classes:
            print(f"{class_name:<30} {metrics['count']:>7} "
                  f"{metrics['dice']:>7.4f} {metrics['iou']:>7.4f} "
                  f"{metrics['precision']:>7.4f} {metrics['recall']:>7.4f}")
        
        print(f"{'='*70}\n")
    
    def save_results(self, save_dir):
        """Save results to JSON and CSV"""
        # Save JSON
        json_path = os.path.join(save_dir, 'evaluation_results.json')
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"📄 Saved JSON results: {json_path}")
        
        # Save per-class CSV
        if self.results['per_class']:
            df_class = pd.DataFrame(self.results['per_class']).T
            csv_path = os.path.join(save_dir, 'per_class_results.csv')
            df_class.to_csv(csv_path)
            print(f"📄 Saved per-class CSV: {csv_path}")
        
        # Save per-sample CSV
        if self.results['per_sample']:
            df_sample = pd.DataFrame(self.results['per_sample'])
            csv_path = os.path.join(save_dir, 'per_sample_results.csv')
            df_sample.to_csv(csv_path, index=False)
            print(f"📄 Saved per-sample CSV: {csv_path}")
    
    def create_visualizations(self, save_dir):
        """Create performance visualizations"""
        print("\n📊 Creating visualizations...")
        
        # 1. Per-class bar chart
        self._plot_per_class_performance(save_dir)
        
        # 2. Metrics distribution
        self._plot_metrics_distribution(save_dir)
        
        # 3. Dice vs IOU scatter
        self._plot_dice_iou_scatter(save_dir)
        
        print("✅ Visualizations created!")
    
    def _plot_per_class_performance(self, save_dir):
        """Plot per-class performance bar chart"""
        if not self.results['per_class']:
            return
        
        # Sort by Dice score
        sorted_classes = sorted(self.results['per_class'].items(), 
                               key=lambda x: x[1]['dice'], reverse=True)
        
        classes = [c[0] for c in sorted_classes]
        dice_scores = [c[1]['dice'] for c in sorted_classes]
        iou_scores = [c[1]['iou'] for c in sorted_classes]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Dice scores
        bars1 = ax1.barh(classes, dice_scores, color='skyblue', edgecolor='navy')
        ax1.set_xlabel('Dice Score', fontsize=12)
        ax1.set_title('Per-Class Dice Score', fontsize=14, fontweight='bold')
        ax1.axvline(x=self.results['overall']['dice'], color='red', 
                   linestyle='--', label=f"Mean: {self.results['overall']['dice']:.4f}")
        ax1.legend()
        ax1.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for bar, score in zip(bars1, dice_scores):
            ax1.text(score + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{score:.3f}', va='center', fontsize=9)
        
        # IOU scores
        bars2 = ax2.barh(classes, iou_scores, color='lightcoral', edgecolor='darkred')
        ax2.set_xlabel('IOU Score', fontsize=12)
        ax2.set_title('Per-Class IOU Score', fontsize=14, fontweight='bold')
        ax2.axvline(x=self.results['overall']['iou'], color='red', 
                   linestyle='--', label=f"Mean: {self.results['overall']['iou']:.4f}")
        ax2.legend()
        ax2.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for bar, score in zip(bars2, iou_scores):
            ax2.text(score + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{score:.3f}', va='center', fontsize=9)
        
        plt.tight_layout()
        save_path = os.path.join(save_dir, 'per_class_performance.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {save_path}")
    
    def _plot_metrics_distribution(self, save_dir):
        """Plot distribution of metrics"""
        if not self.results['per_sample']:
            return
        
        dice_scores = [m['dice'] for m in self.results['per_sample']]
        iou_scores = [m['iou'] for m in self.results['per_sample']]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Dice distribution
        axes[0].hist(dice_scores, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        axes[0].axvline(x=np.mean(dice_scores), color='red', linestyle='--', 
                       linewidth=2, label=f'Mean: {np.mean(dice_scores):.4f}')
        axes[0].set_xlabel('Dice Score', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].set_title('Dice Score Distribution', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # IOU distribution
        axes[1].hist(iou_scores, bins=30, color='lightcoral', edgecolor='black', alpha=0.7)
        axes[1].axvline(x=np.mean(iou_scores), color='red', linestyle='--', 
                       linewidth=2, label=f'Mean: {np.mean(iou_scores):.4f}')
        axes[1].set_xlabel('IOU Score', fontsize=12)
        axes[1].set_ylabel('Frequency', fontsize=12)
        axes[1].set_title('IOU Score Distribution', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(save_dir, 'metrics_distribution.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {save_path}")
    
    def _plot_dice_iou_scatter(self, save_dir):
        """Plot Dice vs IOU scatter with class colors"""
        if not self.results['per_sample']:
            return
        
        # Prepare data
        classes = list(set([m['class'] for m in self.results['per_sample']]))
        colors = plt.cm.tab10(np.linspace(0, 1, len(classes)))
        class_to_color = dict(zip(classes, colors))
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        for class_name in classes:
            class_samples = [m for m in self.results['per_sample'] if m['class'] == class_name]
            dice = [m['dice'] for m in class_samples]
            iou = [m['iou'] for m in class_samples]
            ax.scatter(dice, iou, c=[class_to_color[class_name]], 
                      label=class_name, alpha=0.6, s=50)
        
        # Add diagonal line (theoretical Dice-IOU relationship)
        x = np.linspace(0, 1, 100)
        y = x / (2 - x)  # IOU = Dice / (2 - Dice)
        ax.plot(x, y, 'k--', alpha=0.3, label='Theoretical relationship')
        
        ax.set_xlabel('Dice Score', fontsize=12)
        ax.set_ylabel('IOU Score', fontsize=12)
        ax.set_title('Dice vs IOU Score by Class', fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(save_dir, 'dice_vs_iou.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {save_path}")


def main():
    # Configuration
    model_path = 'checkpoints/medsam2_dermatology_best_norm.pth'
    dataset_path = r'C:\Users\hqgho\OneDrive\Máy tính\Final_Project\MedSAM2\data'
    split = 'val'  # Using validation set (test set has no masks for evaluation)
    save_dir = 'evaluation_results'
    
    print("="*70)
    print("MedSAM2 Model Evaluation")
    print("="*70)
    print(f"Model: {model_path}")
    print(f"Dataset: {dataset_path}")
    print(f"Split: {split}")
    print(f"Output: {save_dir}/")
    print("="*70)
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"\n❌ Error: Model not found at {model_path}")
        return
    
    # Initialize evaluator
    evaluator = MedSAM2Evaluator(model_path)
    
    # Run evaluation
    results = evaluator.evaluate_on_dataset(dataset_path, split=split, save_dir=save_dir)
    
    print(f"\n✅ Evaluation complete! Results saved to: {save_dir}/")
    print("\nGenerated files:")
    print("  - evaluation_results.json (full results)")
    print("  - per_class_results.csv (per-class metrics)")
    print("  - per_sample_results.csv (per-sample metrics)")
    print("  - per_class_performance.png (bar charts)")
    print("  - metrics_distribution.png (histograms)")
    print("  - dice_vs_iou.png (scatter plot)")


if __name__ == '__main__':
    main()
