"""
MedSAM2 Preprocessing Pipeline - Works Like YOLO Pipeline

Filters augmented images before conversion to remove low-quality samples.
Just run this file directly - no command-line arguments needed!

Author: MedSAM2-Dermatology Team
Date: November 2025
"""

import json
import numpy as np
from pathlib import Path
from PIL import Image
import cv2
import shutil
import os

IS_KAGGLE = '/kaggle/' in os.getcwd()

if IS_KAGGLE:
    INPUT_ROOT = '/kaggle/input/yolo11-dataset/tiling_dataset/tiling_dataset_split'
    OUTPUT_ROOT = '/kaggle/working/filtered_data'
else:
    INPUT_ROOT = r'd:\Đồ án( final battle)\Yolo11 model\data\processed\Augment_dataset_split'
    OUTPUT_ROOT = r'd:\Đồ án( final battle)\MedSAM2-Dermatology\data\filtered_data'

# Which splits to process
SPLITS = ['train', 'val']  # Skip test - only need train/val for training

# Quality thresholds (matches YOLO min_area=200)
MIN_PIXELS = 200              # Minimum lesion area in pixels
MIN_PERCENTAGE = 0.5          # Minimum lesion area as % of image
MIN_BBOX_SIZE = 20            # Minimum bounding box width/height
MAX_ASPECT_RATIO = 20.0       # Maximum aspect ratio (filters very thin boxes)
CHECK_ASPECT_RATIO = True     # Enable aspect ratio check

# Class definitions (must match YOLO)
CLASS_NAMES = [
    'Acne', 'Actinic_Keratosis', 'Drug_Eruption', 'Eczema', 'Normal',
    'Psoriasis', 'Rosacea', 'Seborrh_Keratoses', 'Sun_Sunlight_Damage', 
    'Tinea', 'Warts'
]

print("=" * 70)
print("🔍 MEDSAM2 PREPROCESSING - QUALITY FILTERING")
print("=" * 70)
print(f"Environment: {'KAGGLE' if IS_KAGGLE else 'LOCAL'}")
print(f"📂 Input:    {INPUT_ROOT}")
print(f"📦 Output:   {OUTPUT_ROOT}")
print(f"🏷️  Splits:   {', '.join(SPLITS)}")
print(f"🧬 Classes:  {len(CLASS_NAMES)} classes")
print(f"\n📏 Filter Criteria:")
print(f"   - Min pixels:     {MIN_PIXELS}")
print(f"   - Min percentage: {MIN_PERCENTAGE}%")
print(f"   - Min bbox size:  {MIN_BBOX_SIZE}px")
if CHECK_ASPECT_RATIO:
    print(f"   - Max aspect:     {MAX_ASPECT_RATIO}")
print("=" * 70 + "\n")

def analyze_mask_quality(json_path, image_path):
    """
    Analyze LabelMe annotation quality - same criteria as YOLO preprocessing
    Returns: (is_valid, reason, metrics_dict)
    """
    
    # Load JSON annotation
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        return False, f'JSON error: {e}', {}
    
    # Get image dimensions
    if Path(image_path).exists():
        img = Image.open(image_path)
        img_width, img_height = img.size
        img.close()
    else:
        img_width = data.get('imageWidth', 0)
        img_height = data.get('imageHeight', 0)
    
    if img_width == 0 or img_height == 0:
        return False, 'Cannot determine dimensions', {}
    
    total_image_area = img_width * img_height
    
    # Extract shapes (polygon annotations)
    shapes = data.get('shapes', [])
    if not shapes:
        return False, 'No annotations', {}
    
    # Calculate total lesion area from all polygons
    total_lesion_area = 0
    all_points = []
    
    for shape in shapes:
        points = shape.get('points', [])
        if not points or len(points) < 3:
            continue
        
        # Convert to numpy for contour area calculation
        points_array = np.array(points, dtype=np.int32)
        all_points.extend(points)
        
        # Calculate area using OpenCV contourArea (same as YOLO)
        contour_area = cv2.contourArea(points_array)
        total_lesion_area += contour_area
    
    # ========================================
    # CHECK 1: Minimum Pixels (matches YOLO)
    # ========================================
    if total_lesion_area < MIN_PIXELS:
        return False, f'Too small: {total_lesion_area:.0f}px < {MIN_PIXELS}px', {
            'lesion_area': total_lesion_area,
            'percentage': (total_lesion_area / total_image_area) * 100
        }
    
    # ========================================
    # CHECK 2: Minimum Percentage
    # ========================================
    area_percentage = (total_lesion_area / total_image_area) * 100
    if area_percentage < MIN_PERCENTAGE:
        return False, f'Only {area_percentage:.2f}% of image', {
            'lesion_area': total_lesion_area,
            'percentage': area_percentage
        }
    
    # ========================================
    # CHECK 3: Bounding Box Size
    # ========================================
    if all_points:
        all_points_array = np.array(all_points, dtype=np.int32)
        x_coords = all_points_array[:, 0]
        y_coords = all_points_array[:, 1]
        
        bbox_width = x_coords.max() - x_coords.min()
        bbox_height = y_coords.max() - y_coords.min()
        
        if bbox_width < MIN_BBOX_SIZE or bbox_height < MIN_BBOX_SIZE:
            return False, f'BBox too small: {bbox_width}x{bbox_height}', {
                'bbox_width': bbox_width,
                'bbox_height': bbox_height
            }
        
        # ========================================
        # CHECK 4: Aspect Ratio (optional)
        # ========================================
        if CHECK_ASPECT_RATIO:
            aspect_ratio = max(bbox_width, bbox_height) / max(min(bbox_width, bbox_height), 1)
            if aspect_ratio > MAX_ASPECT_RATIO:
                return False, f'Aspect {aspect_ratio:.1f} too extreme', {
                    'aspect_ratio': aspect_ratio
                }
    
    # ✅ Passed all checks
    return True, 'Valid', {
        'lesion_area': total_lesion_area,
        'percentage': area_percentage
    }


def detect_structure(class_folder):
    """
    Detect if class folder has:
    - 'json_flat': JSON + images directly (LabelMe format)
    - 'mask_nested': images/ and masks/ subfolders
    """
    class_items = [item.name for item in class_folder.iterdir()]
    
    # Check for nested structure (images/ and masks/ folders)
    if 'images' in class_items and 'masks' in class_items:
        return 'mask_nested'
    
    # Check for JSON files (LabelMe format)
    if any(item.endswith('.json') for item in class_items):
        return 'json_flat'
    
    return 'unknown'


def analyze_mask_from_file(mask_path):
    """
    Analyze pre-existing mask file quality (for nested structure)
    Returns: (is_valid, reason, metrics_dict)
    """
    try:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return False, 'Cannot read mask', {}
        
        img_height, img_width = mask.shape
        total_image_area = img_width * img_height
        
        # Find contours
        _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return False, 'No contours found', {}
        
        # Calculate total lesion area
        total_lesion_area = sum(cv2.contourArea(c) for c in contours)
        
        # CHECK 1: Minimum pixels
        if total_lesion_area < MIN_PIXELS:
            return False, f'Too small: {total_lesion_area:.0f}px < {MIN_PIXELS}px', {
                'lesion_area': total_lesion_area
            }
        
        # CHECK 2: Minimum percentage
        percentage = (total_lesion_area / total_image_area) * 100
        if percentage < MIN_PERCENTAGE:
            return False, f'Only {percentage:.2f}% of image', {
                'lesion_area': total_lesion_area,
                'percentage': percentage
            }
        
        # CHECK 3: Bounding box
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        if w < MIN_BBOX_SIZE or h < MIN_BBOX_SIZE:
            return False, f'BBox too small: {w}x{h}', {
                'bbox_width': w,
                'bbox_height': h
            }
        
        # CHECK 4: Aspect ratio
        if CHECK_ASPECT_RATIO:
            aspect_ratio = max(w, h) / max(min(w, h), 1)
            if aspect_ratio > MAX_ASPECT_RATIO:
                return False, f'Aspect {aspect_ratio:.1f} too extreme', {
                    'aspect_ratio': aspect_ratio
                }
        
        return True, 'Valid', {
            'lesion_area': total_lesion_area,
            'percentage': percentage
        }
    
    except Exception as e:
        return False, f'Error: {e}', {}

def filter_dataset():
    """Process all splits and filter low-quality images - handles BOTH structures"""
    
    print("=" * 70)
    print("🚀 STARTING QUALITY FILTERING")
    print("=" * 70)
    
    stats = {
        'total': 0,
        'passed': 0,
        'failed': 0
    }
    
    rejected_files = []
    
    for split in SPLITS:
        print(f"\n📂 Processing {split.upper()} split...")
        
        split_input = Path(INPUT_ROOT) / split
        split_output = Path(OUTPUT_ROOT) / split
        
        if not split_input.exists():
            print(f"   ⚠️  Not found: {split_input}")
            continue
        
        # Create output directory
        split_output.mkdir(parents=True, exist_ok=True)
        
        # Find class folders
        class_folders = [d for d in split_input.iterdir() 
                        if d.is_dir() and d.name in CLASS_NAMES]
        
        print(f"   Found {len(class_folders)} class folders")
        
        for class_folder in class_folders:
            class_name = class_folder.name
            output_class_folder = split_output / class_name
            output_class_folder.mkdir(exist_ok=True)
            
            # Auto-detect structure
            structure = detect_structure(class_folder)
            
            valid_count = 0
            rejected_count = 0
            
            # ========================================
            # PROCESS JSON_FLAT STRUCTURE
            # ========================================
            if structure == 'json_flat':
                json_files = list(class_folder.glob('*.json'))
                
                for json_file in json_files:
                    stats['total'] += 1
                    
                    # Find corresponding image
                    image_name = json_file.stem
                    image_file = None
                    for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                        potential_img = class_folder / f"{image_name}{ext}"
                        if potential_img.exists():
                            image_file = potential_img
                            break
                    
                    if not image_file:
                        rejected_files.append({
                            'split': split,
                            'class': class_name,
                            'file': json_file.name,
                            'reason': 'No image found'
                        })
                        stats['failed'] += 1
                        rejected_count += 1
                        continue
                    
                    # Analyze quality from JSON
                    is_valid, reason, metrics = analyze_mask_quality(json_file, image_file)
                    
                    if is_valid:
                        # ✅ Copy valid files
                        shutil.copy2(json_file, output_class_folder / json_file.name)
                        shutil.copy2(image_file, output_class_folder / image_file.name)
                        stats['passed'] += 1
                        valid_count += 1
                    else:
                        # ❌ Track rejected files
                        rejected_files.append({
                            'split': split,
                            'class': class_name,
                            'file': json_file.name,
                            'reason': reason,
                            'metrics': metrics
                        })
                        stats['failed'] += 1
                        rejected_count += 1
            
            # ========================================
            # PROCESS MASK_NESTED STRUCTURE
            # ========================================
            elif structure == 'mask_nested':
                images_dir = class_folder / 'images'
                masks_dir = class_folder / 'masks'
                
                # Create nested output structure
                output_images_dir = output_class_folder / 'images'
                output_masks_dir = output_class_folder / 'masks'
                output_images_dir.mkdir(exist_ok=True)
                output_masks_dir.mkdir(exist_ok=True)
                
                # Process all images
                image_files = list(images_dir.glob('*')) if images_dir.exists() else []
                image_files = [f for f in image_files if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
                
                for image_file in image_files:
                    stats['total'] += 1
                    
                    # Find corresponding mask
                    mask_name = image_file.stem + '.png'
                    mask_file = masks_dir / mask_name
                    
                    if not mask_file.exists():
                        # Try .PNG extension
                        mask_file = masks_dir / (image_file.stem + '.PNG')
                    
                    if not mask_file.exists():
                        rejected_files.append({
                            'split': split,
                            'class': class_name,
                            'file': image_file.name,
                            'reason': 'No mask found'
                        })
                        stats['failed'] += 1
                        rejected_count += 1
                        continue
                    
                    # Analyze quality from mask file
                    is_valid, reason, metrics = analyze_mask_from_file(mask_file)
                    
                    if is_valid:
                        # ✅ Copy valid files
                        shutil.copy2(image_file, output_images_dir / image_file.name)
                        shutil.copy2(mask_file, output_masks_dir / mask_file.name)
                        stats['passed'] += 1
                        valid_count += 1
                    else:
                        # ❌ Track rejected files
                        rejected_files.append({
                            'split': split,
                            'class': class_name,
                            'file': image_file.name,
                            'reason': reason,
                            'metrics': metrics
                        })
                        stats['failed'] += 1
                        rejected_count += 1
            
            else:
                print(f"   ⚠️  {class_name}: Unknown structure, skipping...")
                continue
            
            # Print class summary
            if valid_count > 0 or rejected_count > 0:
                structure_label = "JSON" if structure == 'json_flat' else "MASK"
                print(f"   {class_name:<20} [{structure_label}] ✅ {valid_count:>4} valid  ❌ {rejected_count:>4} rejected")
    
    # Save rejected files list
    if rejected_files:
        rejected_path = Path(OUTPUT_ROOT) / 'rejected_files.json'
        with open(rejected_path, 'w', encoding='utf-8') as f:
            json.dump(rejected_files, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Rejected files saved: {rejected_path}")
    
    # Print final statistics
    print("\n" + "=" * 70)
    print("🎉 FILTERING COMPLETE!")
    print("=" * 70)
    print(f"   Total checked: {stats['total']:,}")
    print(f"   ✅ Passed:     {stats['passed']:,} ({stats['passed']/max(stats['total'],1)*100:.1f}%)")
    print(f"   ❌ Rejected:   {stats['failed']:,} ({stats['failed']/max(stats['total'],1)*100:.1f}%)")
    print(f"   📁 Output:     {OUTPUT_ROOT}")
    print("=" * 70)
    
    return stats, rejected_files


if __name__ == '__main__':
    # Verify input exists
    if not Path(INPUT_ROOT).exists():
        print(f"❌ ERROR: Input path not found!")
        print(f"   Expected: {INPUT_ROOT}")
        print(f"\n💡 Please check your paths and try again.")
    else:
        print(f"✅ Input verified: {INPUT_ROOT}\n")
        
        # Run filtering
        stats, rejected = filter_dataset()
        
        print("\n✅ Preprocessing complete! Ready for conversion.")
