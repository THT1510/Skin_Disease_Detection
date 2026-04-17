"""
Convert LabelMe JSON annotations to organized image/mask structure

Input structure:
    output/
        train/
            Acne/
                image1.jpeg + image1.json
                image2.jpeg + image2.json
            Eczema/
                ...

Output structure:
    data/
        train/
            images/
                Acne/
                    image1.jpg
                    image2.jpg
                Eczema/
                    ...
            masks/
                Acne/
                    image1.png (binary mask: 0=background, 255=lesion)
                    image2.png
                Eczema/
                    ...
        val/
            images/
                Acne/
                    ...
            masks/
                Acne/
                    ...
"""

import os
import json
import shutil
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path
from tqdm import tqdm
import argparse


def load_labelme_json(json_path):
    """Load LabelMe JSON annotation file"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def polygon_to_mask(polygon_points, image_height, image_width):
    """Convert polygon points to binary mask"""
    mask = Image.new('L', (image_width, image_height), 0)
    
    if len(polygon_points) > 0:
        # Convert points to tuples
        polygon = [(float(p[0]), float(p[1])) for p in polygon_points]
        ImageDraw.Draw(mask).polygon(polygon, outline=255, fill=255)
    
    return mask


def process_class_folder(input_class_folder, output_images_folder, output_masks_folder, class_name):
    """
    Process all files in a class folder
    
    Args:
        input_class_folder: Path to input class folder (e.g., output/train/Acne)
        output_images_folder: Path to output images folder (e.g., data/train/images/Acne)
        output_masks_folder: Path to output masks folder (e.g., data/train/masks/Acne)
        class_name: Name of the class (e.g., "Acne")
    
    Returns:
        processed_count: Number of successfully processed files
        skipped_count: Number of skipped files
    """
    input_path = Path(input_class_folder)
    output_img_path = Path(output_images_folder)
    output_mask_path = Path(output_masks_folder)
    
    # Create output directories
    output_img_path.mkdir(parents=True, exist_ok=True)
    output_mask_path.mkdir(parents=True, exist_ok=True)
    
    # Find all JSON files
    json_files = list(input_path.glob('*.json'))
    
    processed = 0
    skipped = 0
    
    for json_file in tqdm(json_files, desc=f"  {class_name}", leave=False):
        try:
            # Find corresponding image file
            image_name = json_file.stem
            
            # Try different image extensions
            image_file = None
            for ext in ['.jpeg', '.jpg', '.png', '.JPEG', '.JPG', '.PNG']:
                potential_image = input_path / f"{image_name}{ext}"
                if potential_image.exists():
                    image_file = potential_image
                    break
            
            if image_file is None:
                print(f"Warning: Image not found for {json_file.name}")
                skipped += 1
                continue
            
            # Load annotation
            annotation = load_labelme_json(str(json_file))
            
            # Load image
            image = Image.open(str(image_file)).convert('RGB')
            
            image_height = annotation['imageHeight']
            image_width = annotation['imageWidth']
            
            # Create combined mask from all shapes
            combined_mask = Image.new('L', (image_width, image_height), 0)
            
            has_annotation = False
            for shape in annotation['shapes']:
                if shape['shape_type'] == 'polygon':
                    points = shape['points']
                    shape_mask = polygon_to_mask(points, image_height, image_width)
                    
                    # Combine masks using maximum
                    combined_mask = Image.fromarray(
                        np.maximum(np.array(combined_mask), np.array(shape_mask))
                    )
                    has_annotation = True
            
            if not has_annotation:
                print(f"Warning: No annotations found in {json_file.name}")
                skipped += 1
                continue
            
            # Check if mask is empty
            mask_array = np.array(combined_mask)
            if mask_array.sum() == 0:
                print(f"Warning: Empty mask for {json_file.name}")
                skipped += 1
                continue
            
            # Save image (copy to images folder with .jpg extension)
            output_image_path = output_img_path / f"{image_name}.jpg"
            image.save(str(output_image_path), 'JPEG', quality=95)
            
            # Save mask (as PNG with .png extension)
            output_mask_path_file = output_mask_path / f"{image_name}.png"
            combined_mask.save(str(output_mask_path_file), 'PNG')
            
            processed += 1
            
        except Exception as e:
            print(f"Error processing {json_file.name}: {str(e)}")
            skipped += 1
            continue
    
    return processed, skipped


def convert_dataset_split(input_root, output_root, split='train'):
    """
    Convert a dataset split (train/val/test)
    
    Args:
        input_root: Path to input folder (e.g., output/train)
        output_root: Path to output folder (e.g., data/train)
        split: 'train', 'val', or 'test'
    """
    input_path = Path(input_root)
    output_path = Path(output_root)
    
    # Create main output directories
    output_images_root = output_path / 'images'
    output_masks_root = output_path / 'masks'
    
    output_images_root.mkdir(parents=True, exist_ok=True)
    output_masks_root.mkdir(parents=True, exist_ok=True)
    
    # Get all class folders
    class_folders = [d for d in input_path.iterdir() if d.is_dir()]
    
    print(f"\n{'='*70}")
    print(f"Converting {split.upper()} dataset")
    print(f"{'='*70}")
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"  - images/ (will contain {len(class_folders)} class folders)")
    print(f"  - masks/  (will contain {len(class_folders)} class folders)")
    print(f"{'='*70}\n")
    
    total_processed = 0
    total_skipped = 0
    
    class_stats = {}
    
    # Process each class folder
    for class_folder in tqdm(class_folders, desc=f"Processing {split} classes"):
        class_name = class_folder.name
        
        # Create class-specific output folders
        output_images_class = output_images_root / class_name
        output_masks_class = output_masks_root / class_name
        
        # Process this class
        processed, skipped = process_class_folder(
            class_folder,
            output_images_class,
            output_masks_class,
            class_name
        )
        
        class_stats[class_name] = {
            'processed': processed,
            'skipped': skipped
        }
        
        total_processed += processed
        total_skipped += skipped
    
    # Print statistics
    print(f"\n{'='*70}")
    print(f"{split.upper()} Conversion Summary")
    print(f"{'='*70}")
    print(f"Total processed: {total_processed}")
    print(f"Total skipped:   {total_skipped}")
    print(f"\nPer-class statistics:")
    print(f"{'Class':<30} {'Processed':>10} {'Skipped':>10}")
    print(f"{'-'*70}")
    for class_name, stats in sorted(class_stats.items()):
        print(f"{class_name:<30} {stats['processed']:>10} {stats['skipped']:>10}")
    print(f"{'='*70}\n")
    
    return total_processed, total_skipped, class_stats


def verify_structure(output_root, split='train'):
    """Verify the output structure is correct"""
    output_path = Path(output_root)
    images_path = output_path / 'images'
    masks_path = output_path / 'masks'
    
    print(f"\nVerifying {split} structure...")
    print(f"{'='*70}")
    
    if not images_path.exists():
        print(f"ERROR: {images_path} does not exist!")
        return False
    
    if not masks_path.exists():
        print(f"ERROR: {masks_path} does not exist!")
        return False
    
    # Get class folders
    image_classes = sorted([d.name for d in images_path.iterdir() if d.is_dir()])
    mask_classes = sorted([d.name for d in masks_path.iterdir() if d.is_dir()])
    
    if image_classes != mask_classes:
        print(f"ERROR: Class folders mismatch!")
        print(f"  Images: {image_classes}")
        print(f"  Masks:  {mask_classes}")
        return False
    
    print(f"✓ Found {len(image_classes)} classes in both images/ and masks/")
    print(f"\nClasses: {', '.join(image_classes)}")
    
    # Check each class
    print(f"\nPer-class file counts:")
    print(f"{'Class':<30} {'Images':>10} {'Masks':>10} {'Match':>10}")
    print(f"{'-'*70}")
    
    all_match = True
    for class_name in image_classes:
        img_files = list((images_path / class_name).glob('*.jpg'))
        mask_files = list((masks_path / class_name).glob('*.png'))
        
        img_count = len(img_files)
        mask_count = len(mask_files)
        
        match = '✓' if img_count == mask_count else '✗'
        if img_count != mask_count:
            all_match = False
        
        print(f"{class_name:<30} {img_count:>10} {mask_count:>10} {match:>10}")
    
    print(f"{'='*70}")
    
    if all_match:
        print(f"✓ All classes have matching image and mask counts")
        
        # Check a sample
        sample_class = image_classes[0]
        sample_img = list((images_path / sample_class).glob('*.jpg'))[0]
        sample_mask = (masks_path / sample_class / (sample_img.stem + '.png'))
        
        if sample_mask.exists():
            print(f"\nSample verification ({sample_class}):")
            print(f"  Image: {sample_img.name}")
            print(f"  Mask:  {sample_mask.name}")
            
            img = Image.open(sample_img)
            mask = Image.open(sample_mask)
            
            print(f"  Image size: {img.size}")
            print(f"  Mask size:  {mask.size}")
            print(f"  Mask range: {np.array(mask).min()}-{np.array(mask).max()}")
            print(f"  ✓ Valid structure")
        
        return True
    else:
        print(f"✗ Some classes have mismatched counts!")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Convert LabelMe JSON to organized image/mask structure'
    )
    parser.add_argument(
        '--input_root',
        type=str,
        default=r'C:\Users\hqgho\OneDrive\Máy tính\Final_Project\output',
        help='Root directory containing train/val/test folders'
    )
    parser.add_argument(
        '--output_root',
        type=str,
        default=r'C:\Users\hqgho\OneDrive\Máy tính\Final_Project\aug_data',
        help='Output directory'
    )
    parser.add_argument(
        '--splits',
        nargs='+',
        default=['train', 'val', 'test'],
        help='Dataset splits to convert (default: train val test)'
    )
    
    args, unknown = parser.parse_known_args()
    
    print("\n" + "="*70)
    print("LabelMe JSON to Image/Mask Structure Converter")
    print("="*70)
    
    total_all = 0
    skipped_all = 0
    
    # Convert each split
    for split in args.splits:
        input_dir = os.path.join(args.input_root, split)
        output_dir = os.path.join(args.output_root, split)
        
        if not os.path.exists(input_dir):
            print(f"\nWarning: {input_dir} does not exist. Skipping {split}.")
            continue
        
        processed, skipped, _ = convert_dataset_split(input_dir, output_dir, split)
        total_all += processed
        skipped_all += skipped
        
        # Verify the output
        verify_structure(output_dir, split)
    
    # Final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print(f"Total files processed: {total_all}")
    print(f"Total files skipped:   {skipped_all}")
    print(f"\nOutput structure:")
    for split in args.splits:
        split_dir = os.path.join(args.output_root, split)
        if os.path.exists(split_dir):
            images_dir = os.path.join(split_dir, 'images')
            masks_dir = os.path.join(split_dir, 'masks')
            
            if os.path.exists(images_dir):
                num_classes = len([d for d in Path(images_dir).iterdir() if d.is_dir()])
                total_images = len(list(Path(images_dir).rglob('*.jpg')))
                total_masks = len(list(Path(masks_dir).rglob('*.png')))
                
                print(f"\n{split}/")
                print(f"  images/  - {num_classes} classes, {total_images} images (.jpg)")
                print(f"  masks/   - {num_classes} classes, {total_masks} masks (.png)")
    
    print("="*70)
    print("\n✓ Conversion complete!")
    print("\nOutput structure:")
    print("  data/")
    print("    train/")
    print("      images/")
    print("        Acne/")
    print("          image1.jpg")
    print("          image2.jpg")
    print("        Eczema/")
    print("          ...")
    print("      masks/")
    print("        Acne/")
    print("          image1.png (binary: 0=background, 255=lesion)")
    print("          image2.png")
    print("        Eczema/")
    print("          ...")
    print("    val/")
    print("      images/...")
    print("      masks/...")
    print("\nNext step: Update your training script to load from this structure!")


if __name__ == '__main__':
    main()
