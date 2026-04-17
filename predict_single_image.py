"""
Predict and visualize segmentation for a single image or directory of images.

Usage:
    python predict_single_image.py --image path/to/image.jpg
    python predict_single_image.py --dir path/to/images/
"""

import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import torch.nn.functional as F
from hydra import initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from sam2.build_sam import build_sam2


class MedSAM2Predictor:
    """Single image prediction with MedSAM2"""
    
    def __init__(self, model_path, config_name='sam2.1_hiera_t512'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 Using device: {self.device}")
        
        # Build and load model
        self.model = self.build_model(config_name)
        self.load_checkpoint(model_path)
        self.model.eval()
    
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
        # Load image
        image = Image.open(image_path).convert('RGB')
        original_size = image.size
        
        # Resize
        image = image.resize((target_size, target_size), Image.BILINEAR)
        image_array = np.array(image).astype(np.float32) / 255.0
        
        # To tensor
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0)
        
        return image_tensor.to(self.device), image_array, original_size
    
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
        
        # Prompt embeddings (no prompts = automatic segmentation)
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
    
    def visualize_prediction(self, image_array, pred_mask, save_path=None, threshold=0.5):
        """Visualize prediction with overlay"""
        # Convert prediction to binary mask
        pred_binary = (pred_mask.cpu().numpy().squeeze() > threshold).astype(np.uint8)
        
        # Create figure with 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Original image
        axes[0].imshow(image_array)
        axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        # Predicted mask
        axes[1].imshow(pred_binary, cmap='gray')
        axes[1].set_title('Predicted Mask', fontsize=14, fontweight='bold')
        axes[1].axis('off')
        
        # Overlay
        axes[2].imshow(image_array)
        axes[2].imshow(pred_binary, cmap='Reds', alpha=0.5)
        axes[2].set_title('Overlay (Red = Lesion)', fontsize=14, fontweight='bold')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"💾 Saved: {save_path}")
        
        plt.show()
        plt.close()
    
    def predict_image(self, image_path, output_dir='predictions', threshold=0.5):
        """Predict single image and save result"""
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n{'='*70}")
        print(f"Processing: {image_path}")
        print(f"{'='*70}")
        
        # Preprocess
        image_tensor, image_array, original_size = self.preprocess_image(image_path)
        
        # Predict
        pred_mask = self.predict(image_tensor)
        
        # Calculate mask statistics
        pred_binary = (pred_mask > threshold).float()
        lesion_pixels = pred_binary.sum().item()
        total_pixels = pred_binary.numel()
        lesion_percentage = (lesion_pixels / total_pixels) * 100
        
        print(f"✅ Prediction complete!")
        print(f"   Lesion area: {lesion_percentage:.2f}% of image")
        print(f"   Lesion pixels: {int(lesion_pixels):,} / {total_pixels:,}")
        
        # Save visualization
        image_name = Path(image_path).stem
        save_path = os.path.join(output_dir, f'{image_name}_prediction.png')
        self.visualize_prediction(image_array, pred_mask, save_path, threshold)
        
        # Also save mask as image
        mask_save_path = os.path.join(output_dir, f'{image_name}_mask.png')
        mask_image = Image.fromarray((pred_binary.cpu().numpy().squeeze() * 255).astype(np.uint8))
        mask_image.save(mask_save_path)
        print(f"💾 Saved mask: {mask_save_path}")
        
        return pred_mask, lesion_percentage
    
    def predict_directory(self, dir_path, output_dir='predictions', threshold=0.5, max_images=None):
        """Predict all images in a directory"""
        dir_path = Path(dir_path)
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(dir_path.glob(f'*{ext}'))
            image_files.extend(dir_path.glob(f'*{ext.upper()}'))
        
        if len(image_files) == 0:
            print(f"❌ No images found in {dir_path}")
            return
        
        if max_images:
            image_files = image_files[:max_images]
        
        print(f"\n{'='*70}")
        print(f"Found {len(image_files)} images to process")
        print(f"{'='*70}")
        
        results = []
        
        for idx, image_path in enumerate(image_files, 1):
            print(f"\n[{idx}/{len(image_files)}]")
            try:
                pred_mask, lesion_pct = self.predict_image(str(image_path), output_dir, threshold)
                results.append({
                    'filename': image_path.name,
                    'lesion_percentage': lesion_pct
                })
            except Exception as e:
                print(f"❌ Error processing {image_path.name}: {e}")
        
        # Print summary
        print(f"\n{'='*70}")
        print("SUMMARY")
        print(f"{'='*70}")
        print(f"Total images processed: {len(results)}")
        if results:
            avg_lesion = np.mean([r['lesion_percentage'] for r in results])
            print(f"Average lesion area: {avg_lesion:.2f}%")
        print(f"Results saved to: {output_dir}/")
        print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(description='Predict segmentation with MedSAM2')
    parser.add_argument('--image', type=str, help='Path to single image')
    parser.add_argument('--dir', type=str, help='Path to directory of images')
    parser.add_argument('--model', type=str, default='checkpoints/medsam2_dermatology_best_aug2.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default='predictions',
                       help='Output directory for predictions')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Threshold for binary mask (0-1)')
    parser.add_argument('--max_images', type=int, default=None,
                       help='Maximum number of images to process')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.image and not args.dir:
        print("❌ Error: Please provide either --image or --dir")
        print("\nExamples:")
        print("  python predict_single_image.py --image data/test/Acne/acne-cystic-1.jpeg")
        print("  python predict_single_image.py --dir data/test/Acne/ --max_images 10")
        return
    
    # Check model exists
    if not os.path.exists(args.model):
        print(f"❌ Error: Model not found at {args.model}")
        return
    
    print("="*70)
    print("MedSAM2 Single Image Prediction")
    print("="*70)
    print(f"Model: {args.model}")
    print(f"Threshold: {args.threshold}")
    print(f"Output: {args.output}/")
    print("="*70)
    
    # Initialize predictor
    predictor = MedSAM2Predictor(args.model)
    
    # Predict
    if args.image:
        predictor.predict_image(args.image, args.output, args.threshold)
    elif args.dir:
        predictor.predict_directory(args.dir, args.output, args.threshold, args.max_images)
    
    print("\n✅ Done!")


if __name__ == '__main__':
    main()
