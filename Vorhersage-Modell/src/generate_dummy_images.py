"""Generate dummy game screenshots for multimodal training.

Each dummy image simulates a game screen with random visual noise/patterns.
This allows testing the multimodal architecture without real screenshot data.

Usage:
    python3 src/generate_dummy_images.py --output data/dummy_screenshots --n 100
"""
import argparse
import os
import numpy as np
from PIL import Image, ImageDraw
import json


def generate_dummy_image(width=640, height=480, seed=None):
    """Generate a synthetic 'game screenshot' with random patterns/noise.
    
    Args:
        width, height: Image dimensions
        seed: Random seed for reproducibility
    
    Returns:
        PIL Image object
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Create base image (dark blue like game background)
    img = Image.new('RGB', (width, height), color=(20, 40, 80))
    draw = ImageDraw.Draw(img)
    
    # Add random rectangles simulating UI elements / terrain
    for _ in range(5):
        x1 = np.random.randint(0, width)
        y1 = np.random.randint(0, height)
        x2 = x1 + np.random.randint(50, 200)
        y2 = y1 + np.random.randint(50, 200)
        color = tuple(np.random.randint(50, 200, 3))
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
    
    # Add random circles (simulating player positions)
    for _ in range(8):
        cx = np.random.randint(50, width-50)
        cy = np.random.randint(50, height-50)
        r = np.random.randint(5, 20)
        color = tuple(np.random.randint(100, 255, 3))
        draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=color)
    
    # Add noise texture
    pixels = np.array(img)
    noise = np.random.randint(-20, 20, pixels.shape)
    pixels = np.clip(pixels + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(pixels)
    
    return img


def main(args):
    os.makedirs(args.output, exist_ok=True)
    
    # Generate images and create metadata CSV
    metadata = []
    for i in range(args.n):
        match_id = f"demo_{i // 10:03d}"
        frame_id = i % 10
        
        img = generate_dummy_image(seed=i)
        img_path = os.path.join(args.output, f"{match_id}_frame_{frame_id}.jpg")
        img.save(img_path, quality=85)
        
        metadata.append({
            'match_id': match_id,
            'frame_id': frame_id,
            'image_path': img_path
        })
    
    # Save metadata as JSON lines for easy loading
    meta_path = os.path.join(args.output, 'metadata.jsonl')
    with open(meta_path, 'w') as f:
        for m in metadata:
            f.write(json.dumps(m) + '\n')
    
    print(f'Generated {args.n} dummy images in {args.output}')
    print(f'Metadata saved to {meta_path}')
    print(f'Image dimensions: 640x480')
    print(f'Sample: {metadata[0]}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', '-o', default='data/dummy_screenshots', help='Output directory for images')
    parser.add_argument('--n', type=int, default=100, help='Number of images to generate')
    args = parser.parse_args()
    main(args)
