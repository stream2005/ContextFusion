#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import os
sys.path.append(os.getcwd())
import time
import datetime
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings('ignore')  
import logging
logging.basicConfig(level=logging.CRITICAL)
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from nets.ContextFusion import ContextFusion
from losses import ir_loss, vi_loss, ssim_loss, gra_loss
from utils import H5Dataset
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import random
import numpy as np

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

num_epochs = 120
lr = 1e-4
batch_size = 8
device = 'cuda' if torch.cuda.is_available() else 'cpu'
configs = [
    {
        'name':'ContextFusion',
        'description':'ContextFusion',
        'model_class': ContextFusion,
        'model_name': 'ContextFusion'
    },
]

trainloader = DataLoader(
    H5Dataset(r"dataprocessing/Data/MSRS_train_128_200.h5"),
    batch_size=batch_size, 
    shuffle=True, 
    num_workers=6
)
weight = [1, 1, 10, 100]
torch.backends.cudnn.benchmark = True

def train_single_model(config, model_idx):
    print(f"\n{'='*60}")
    print(f"Model Architecture: {config['model_name']}")
    print(f"{'='*60}")
    
    # Initialize model
    model = config['model_class']().to(device)
    
    # Initialize optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40, 60, 80], gamma=0.5)
    # Create save directories
    save_dir = f"model/{config['name']}"
    tensorboard_dir = f"runs/{config['name']}"
    
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)
    
    # Initialize tensorboard writer
    writer = SummaryWriter(log_dir=tensorboard_dir)
    
    # Training statistics
    step = 0
    running_loss = 0
    prev_time = time.time()
    
    print(f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print(f"Model save directory: {save_dir}")
    print(f"Tensorboard log directory: {tensorboard_dir}")
    
    # Start training process
    for epoch in range(num_epochs):
        model.train()
        
        with tqdm(trainloader, desc=f'{config["description"]} - Epoch {epoch+1}/{num_epochs}', unit='batch') as tepoch:
            for i, (data_IR, data_VIS, index) in enumerate(tepoch):
                data_VIS, data_IR = data_VIS.cuda(), data_IR.cuda()
                
                # Forward propagation
                F = model(data_IR, data_VIS)
                
                # Calculate individual loss components (unweighted for display purposes)
                optimizer.zero_grad()
                loss_ir = ir_loss(F, data_IR)
                loss_vi = vi_loss(F, data_VIS)
                loss_ssim = ssim_loss(F, data_IR, data_VIS)
                loss_gra = gra_loss(F, data_IR, data_VIS)
                # Total weighted loss computation
                loss_total = (
                    weight[0] * loss_ir +
                    weight[1] * loss_vi +
                    weight[2] * loss_ssim +
                    weight[3] * loss_gra
                )
                
                loss_total.backward()
                optimizer.step()
                
                running_loss += loss_total.item()
                tepoch.set_postfix(
                    loss=loss_total.item(),
                    ir=loss_ir.item(),
                    vi=loss_vi.item(),
                    ssim=loss_ssim.item(),
                    gra=loss_gra.item(),
                    time=str(datetime.timedelta(seconds=time.time() - prev_time))
                )
                prev_time = time.time()
                if (i + 1) % 811 == 0:
                    avg_train_loss = running_loss / 811
                    writer.add_scalar('Loss/train', avg_train_loss, step)
                    running_loss = 0.0
                    step += 1
        
        scheduler.step()
        if (epoch + 1) % 1 == 0:
            checkpoint_path = os.path.join(save_dir, f'{config["name"]}_epoch_{epoch+1}.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  ✓ Checkpoint saved: {checkpoint_path}")
            
            # Log metrics to tensorboard
            writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
            writer.add_scalar('Epoch_Loss', running_loss / len(trainloader) if running_loss > 0 else 0, epoch)
    
    # Save final model
    final_model_path = os.path.join(save_dir, f'{config["name"]}_final.pth')
    torch.save(model.state_dict(), final_model_path)
    print(f"  ✓ Final model saved: {final_model_path}")
    
    # Close tensorboard writer
    writer.close()
    
    print(f"Training for {config['description']} completed successfully!")
    return model

def main():
    print(f"Training epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {lr}")
    print(f"Device: {device}")
    
    trained_models = []
    for idx, config in enumerate(configs):
        try:
            model = train_single_model(config, idx)
            trained_models.append((config['name'], model))
        except Exception as e:
            print(f"Error during training model {config['name']}: {e}")
            continue
    
    print(f"\n{'='*60}")
    print(f"Number of successfully trained models: {len(trained_models)}")
    
    if trained_models:
        print("Successfully trained models:")
        for name, _ in trained_models:
            print(f"  ✓ {name}")
    
    print(f"{'='*60}")
    os.system("shutdown -h now")
if __name__ == "__main__":
    main()
