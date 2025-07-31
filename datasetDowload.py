
import os
import urllib.request
import cv2
import numpy as np
import sys
from pathlib import Path

if sys.stdout.encoding != 'utf-8':
    sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)
if sys.stderr.encoding != 'utf-8':
    sys.stderr = open(sys.stderr.fileno(), mode='w', encoding='utf-8', buffering=1)

def print_banner():
    """Print welcome banner"""
    print("="*70)
    print(" TEST MEDIA DOWNLOADER FOR OBJECT DETECTION")
    print("="*70)
    print("This script will download sample images and videos to test your system")
    print()

def download_with_progress(url, filename, description):
    """Download file with progress bar"""
    def progress_hook(block_num, block_size, total_size):
        if total_size > 0:
            percent = min(100, (block_num * block_size * 100) // total_size)
            bar = '█' * (percent // 2) + '░' * (50 - percent // 2)
            print(f"\r  [{bar}] {percent}%", end='', flush=True)
    
    try:
        print(f"\n  Downloading {description}...")
        urllib.request.urlretrieve(url, filename, progress_hook)
        print(f"\n Downloaded: {filename}")
        return True
    except Exception as e:
        print(f"\n Failed to download {filename}: {e}")
        return False

def download_real_test_images():
    """Download real test images from public sources"""
    print("\n Downloading real test images...")
    
    test_images = [
        {
            'url': 'https://images.unsplash.com/photo-1449824913935-59a10b8d2000?w=640&h=480&fit=crop',
            'filename': 'test_images/city_street.jpg',
            'description': 'City street scene'
        },
        {
            'url': 'https://images.unsplash.com/photo-1551963831-b3b1ca40c98e?w=640&h=480&fit=crop',
            'filename': 'test_images/breakfast_table.jpg',
            'description': 'Breakfast table with objects'
        },
        {
            'url': 'https://images.unsplash.com/photo-1574158622682-e40e69881006?w=640&h=480&fit=crop',
            'filename': 'test_images/cat_and_dog.jpg',
            'description': 'Cat and dog photo'
        }
    ]
    
    success_count = 0
    for image_info in test_images:
        try:
            if download_with_progress(image_info['url'], image_info['filename'], image_info['description']):
                success_count += 1
        except Exception as e:
            print(f"  Could not download {image_info['description']}: {e}")
            continue
    
    print(f"Downloaded {success_count} real test images")
    return success_count > 0

def download_sample_videos():
    print("\n Looking for sample videos online...")
    
    sample_videos = [
        {
            'url': 'http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4',
            'filename': 'test_videos/sample_video.mp4',
            'description': 'Sample video (Big Buck Bunny)'
        }
    ]
    
    success_count = 0
    for video_info in sample_videos:
        try:
            if download_with_progress(video_info['url'], video_info['filename'], video_info['description']):
                success_count += 1
        except Exception as e:
            print(f"  Could not download {video_info['description']}: {e}")
            continue
    
    if success_count == 0:
        print("  No online videos downloaded (this is normal)")
    
    return True 

def run_verification_tests():
    """Run tests to verify the downloaded media works"""
    print("\n Running verification tests...")
    
    # Test images
    image_files = [
        'test_images/street_scene.jpg',
        'test_images/indoor_scene.jpg', 
        'test_images/mixed_objects.jpg'
    ]
    
    print("\nTesting images:")
    for img_file in image_files:
        if os.path.exists(img_file):
            try:
                img = cv2.imread(img_file)
                if img is not None:
                    h, w = img.shape[:2]
                    print(f" {img_file} - {w}x{h} pixels")
                else:
                    print(f" {img_file} - Cannot read image")
            except Exception as e:
                print(f" {img_file} - Error: {e}")
        else:
            print(f" {img_file} - File not found")


def main():
    print_banner()
    
    try:
        
        try_real_images = input("\n Try to download real test images from internet? (y/n): ").strip().lower()
        if try_real_images == 'y':
            download_real_test_images()
        

        download_sample_videos()
        
        run_verification_tests()
        
        return True
        
    except Exception as e:
        print(f"\n Error during setup: {e}")
        return False

if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\n SUCCESS! Your test media is ready!")
        else:
            print("\n Setup failed. Please check the errors above.")
    except KeyboardInterrupt:
        print("\n\n Setup interrupted by user.")
    except Exception as e:
        print(f"\n Unexpected error: {e}")