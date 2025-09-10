import sys
import os
import math
import numpy as np
import torch
import torch.nn as nn
import cv2
from tqdm import tqdm
import logging
import threading
import queue
import time
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

logging.basicConfig(level=logging.CRITICAL)

sys.path.append(os.getcwd())
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from utils import image_read_cv2, img_save
from nets.ContextFusion import ContextFusion

# ==================== Path Configuration ====================
path_ir = r"..."
path_vi = r"..."
path_save = r"..."
path_model = r"..."

# ==================== Model Initialization ====================
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = ContextFusion().to(device)


model.load_state_dict(torch.load(path_model))
model.eval()

# ==================== Patch Splitting Functions ====================
def split_into_patches(image, patch_size=128, overlap=64):
    stride = patch_size - overlap
    if len(image.shape) == 2:
        h, w = image.shape
        c = 1
        pad_mode = 'reflect'
    else:
        c, h, w = image.shape
        pad_mode = 'constant'

    # Calculate padding dimensions
    nH = max(0, (h - patch_size + stride) // stride) + 1
    nW = max(0, (w - patch_size + stride) // stride) + 1
    padded_h = (nH - 1) * stride + patch_size
    padded_w = (nW - 1) * stride + patch_size

    pad_h = max(0, padded_h - h)
    pad_w = max(0, padded_w - w)

    # Apply padding to image
    if len(image.shape) == 3:
        padded_image = np.pad(image, ((0, 0), (0, pad_h), (0, pad_w)), mode=pad_mode)
    else:
        padded_image = np.pad(image, ((0, pad_h), (0, pad_w)), mode=pad_mode)

    # Construct image patches
    patches = []
    coordinates = []
    for i in range(0, padded_h - patch_size + 1, stride):
        for j in range(0, padded_w - patch_size + 1, stride):
            if len(image.shape) == 3:
                patch = padded_image[:, i:i+patch_size, j:j+patch_size]
            else:
                patch = padded_image[i:i+patch_size, j:j+patch_size]
            patches.append(patch)
            coordinates.append((i, j))

    return patches, coordinates, (h, w), (padded_h, padded_w)


# ==================== Patch Merging Functions ====================
def merge_patches(patches, coordinates, original_shape, padded_shape, patch_size=128, overlap=64):
    stride = patch_size - overlap
    padded_h, padded_w = padded_shape

    # Create Hanning window
    window_x = torch.hann_window(patch_size, periodic=False, device=device)
    window = torch.outer(window_x, window_x).float()  # [H, W]

    # Initialize merged image and weight map
    merged = torch.zeros((1, padded_h, padded_w), device=device, dtype=torch.float32)
    weight = torch.zeros((1, padded_h, padded_w), device=device, dtype=torch.float32)

    # Convert patches to Tensor
    patches_tensor = torch.from_numpy(np.stack(patches)).to(device)  # [N, H, W]

    # Batch apply window weights and accumulate
    for idx, (i, j) in enumerate(coordinates):
        merged[0, i:i+patch_size, j:j+patch_size] += patches_tensor[idx] * window
        weight[0, i:i+patch_size, j:j+patch_size] += window

    # Safe normalization
    merged = torch.where(weight > 1e-8, merged / weight, torch.zeros_like(merged))

    # Crop back to original size
    merged = merged[0, :original_shape[0], :original_shape[1]]
    return merged.cpu().numpy()


# ==================== Main Image Processing Pipeline ====================
def process_image(ir, vi, model, device):
    ir_np = ir.squeeze()
    vi_np = vi.squeeze()

    patch_size = 128
    overlap = 64

    ir_patches, coords, orig_shape, padded_shape = split_into_patches(ir_np, patch_size, overlap)
    vi_patches, _, _, _ = split_into_patches(vi_np, patch_size, overlap)

    # Batch convert to Tensor
    ir_tensor = torch.from_numpy(np.stack(ir_patches)).unsqueeze(1).to(device)  # [N, 1, H, W]
    vi_tensor = torch.from_numpy(np.stack(vi_patches)).unsqueeze(1).to(device)

    # Batch inference
    with torch.no_grad():
        fused = model(ir_tensor, vi_tensor)  # [N, 1, H, W]
        fused = fused.squeeze(1).cpu().numpy()  # [N, H, W]

    # Merge results
    merged = merge_patches(fused, coords, orig_shape, padded_shape, patch_size, overlap)
    return merged


# ==================== Asynchronous Data Loader ====================
class ImageLoader:
    def __init__(self, path_ir, path_vi, max_queue_size=8):
        self.path_ir = path_ir
        self.path_vi = path_vi
        self.image_list = os.listdir(path_ir)
        self.queue = queue.Queue(maxsize=max_queue_size)
        self.stop_event = threading.Event()
        
    def start_loading(self):
        self.loader_thread = threading.Thread(target=self._load_images)
        self.loader_thread.daemon = True
        self.loader_thread.start()
        
    def _load_images(self):
        for imgname in self.image_list:
            if self.stop_event.is_set():
                break
            try:
                vi_name = imgname.replace("ir", "vi")
                
                # Load infrared image
                IR = image_read_cv2(os.path.join(self.path_ir, imgname), 'GRAY') / 255.0
                
                # Load visible light image
                vi_path = os.path.join(self.path_vi, vi_name)
                vi_image = cv2.imread(vi_path, cv2.IMREAD_UNCHANGED)
                
                if vi_image is not None:
                    self.queue.put((imgname, IR, vi_image))
                else:
                    print(f"Warning: Unable to load visible image: {vi_path}")
            except Exception as e:
                print(f"Warning: Error loading image {imgname}: {e}")
        
        # Send termination signal
        self.queue.put(None)
    
    def get_next(self):
        return self.queue.get()
    
    def stop(self):
        self.stop_event.set()


# ==================== Results saver ====================
class ResultSaver:
    def __init__(self, save_path, max_queue_size=8):
        self.save_path = save_path
        self.queue = queue.Queue(maxsize=max_queue_size)
        self.stop_event = threading.Event()
        
    def start_saving(self):
        self.saver_thread = threading.Thread(target=self._save_results)
        self.saver_thread.daemon = True
        self.saver_thread.start()
        
    def _save_results(self):
        while not self.stop_event.is_set():
            try:
                item = self.queue.get(timeout=1)
                if item is None:  # end signal
                    break
                fused_image, imgname = item
                img_save(fused_image, imgname.split('.')[0], self.save_path)
                self.queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"⚠️ An error occurred while saving the image: {e}")
    
    def save(self, fused_image, imgname):
        self.queue.put((fused_image, imgname))
    
    def stop(self):
        self.queue.put(None)
        self.saver_thread.join()


# ==================== Multi-threaded Processing Function ====================
def process_single_image(args, model, device):
    imgname, IR, vi_image = args
    
    try:
        # Process color or grayscale images
        if vi_image.ndim == 3 and vi_image.shape[2] == 3:
            vi_yuv = cv2.cvtColor(vi_image, cv2.COLOR_BGR2YUV)
            Y, U, V = cv2.split(vi_yuv)
            Y_norm = Y.astype(np.float32) / 255.0
            fused_Y = process_image(IR, Y_norm, model, device)
            fused_Y = (fused_Y * 255).clip(0, 255).astype(np.uint8)
            fused_yuv = cv2.merge([fused_Y, U, V])
            fused_image = cv2.cvtColor(fused_yuv, cv2.COLOR_YUV2BGR)
        else:
            if vi_image.ndim == 3:
                VI = cv2.cvtColor(vi_image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            else:
                VI = vi_image.astype(np.float32) / 255.0
            fused_image = process_image(IR, VI, model, device)
            fused_image = (fused_image * 255).clip(0, 255).astype(np.uint8)
        
        return fused_image, imgname
    except Exception as e:
        print(f"⚠️ An error occurred in processing the image {imgname}: {e}")
        return None, imgname


# ==================== Main Function ====================
def main():
    # Create output directory
    os.makedirs(path_save, exist_ok=True)
    
    # Get image list for progress bar
    image_list = os.listdir(path_ir)
    total_images = len(image_list)
    
    # Initialize asynchronous loader and saver
    loader = ImageLoader(path_ir, path_vi)
    saver = ResultSaver(path_save)
    
    loader.start_loading()
    saver.start_saving()
    
    print("🚀 Starting image processing...")
    start_time = time.time()
    
    # Use tqdm progress bar
    pbar = tqdm(total=total_images, desc="Processing images", unit="it")
    
    # Process images using thread pool
    processed_count = 0
    with ThreadPoolExecutor(max_workers=4) as executor:  # Adjust based on CPU cores
        futures = []
        
        while processed_count < total_images:
            item = loader.get_next()
            if item is None:  # All images loaded
                break
                
            # Submit processing task
            future = executor.submit(process_single_image, item, model, device)
            futures.append(future)
            
            # Check completed tasks and save results
            completed_futures = [f for f in futures if f.done()]
            for future in completed_futures:
                fused_image, imgname = future.result()
                if fused_image is not None:
                    saver.save(fused_image, imgname)
                futures.remove(future)
                processed_count += 1
                pbar.update(1)
        
        # Wait for remaining tasks to complete
        for future in futures:
            fused_image, imgname = future.result()
            if fused_image is not None:
                saver.save(fused_image, imgname)
            processed_count += 1
            pbar.update(1)
    
    pbar.close()
    
    # Wait for all save tasks to complete
    saver.queue.join()
    
    # Cleanup resources
    loader.stop()
    saver.stop()
    
    end_time = time.time()
    total_time = end_time - start_time
    fps = total_images / total_time if total_time > 0 else 0
    
    print(f"✅ Processing completed!")
    print(f"Total images: {total_images}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Processing speed: {fps:.2f} it/s")


if __name__ == "__main__":
    main()
