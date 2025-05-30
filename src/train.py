import argparse
import datetime
import gc
import os
import warnings
from os.path import basename, join, splitext

import numpy as np
import pandas as pd
import torchvision
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
from dotenv import load_dotenv
from sklearn.model_selection import StratifiedKFold
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.audio_processing import process_data
from src.config import Config
from src.dataset import BirdCLEFDatasetFromNPY, collate_fn
from src.utils import calculate_auc, set_seed

warnings.filterwarnings("ignore")


class BirdCLEFModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        taxonomy_df = pd.read_csv(cfg.taxonomy_csv)
        cfg.num_classes = len(taxonomy_df)
        
        self.backbone = timm.create_model(
            cfg.model_name,
            pretrained=cfg.pretrained,
            in_chans=cfg.in_channels,
            drop_rate=cfg.drop_rate,
            drop_path_rate=cfg.drop_path_rate,
        )
        
        if 'efficientnet' in cfg.model_name:
            backbone_out = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
        elif 'resnet' in cfg.model_name:
            backbone_out = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        else:
            backbone_out = self.backbone.get_classifier().in_features
            self.backbone.reset_classifier(0, '')
        
        self.pooling = nn.AdaptiveAvgPool2d(1)
        
        self.feat_dim = backbone_out
        
        self.classifier = nn.Linear(backbone_out, cfg.num_classes)
        
        self.mixup_enabled = hasattr(cfg, 'mixup_alpha') and cfg.mixup_alpha > 0
        if self.mixup_enabled:
            self.mixup_alpha = cfg.mixup_alpha
            
    def forward(self, x, targets=None):
        if self.training and self.mixup_enabled and targets is not None:
            mixed_x, targets_a, targets_b, lam = self.mixup_data(x, targets)
            x = mixed_x
        else:
            targets_a, targets_b, lam = None, None, None
        
        features = self.backbone(x)
        
        if isinstance(features, dict):
            features = features['features']
            
        if len(features.shape) == 4:
            features = self.pooling(features)
            features = features.view(features.size(0), -1)
        
        logits = self.classifier(features)
        
        if self.training and self.mixup_enabled and targets is not None:
            loss = self.mixup_criterion(F.binary_cross_entropy_with_logits, 
                                       logits, targets_a, targets_b, lam)
            return logits, loss
            
        return logits
    
    def mixup_data(self, x, targets):
        """Applies mixup to the data batch"""
        batch_size = x.size(0)

        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)

        indices = torch.randperm(batch_size).to(x.device)

        mixed_x = lam * x + (1 - lam) * x[indices]
        
        return mixed_x, targets, targets[indices], lam
    
    def mixup_criterion(self, criterion, pred, y_a, y_b, lam):
        """Applies mixup to the loss function"""
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def get_optimizer(model, cfg):
  
    if cfg.optimizer == 'Adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay
        )
    elif cfg.optimizer == 'AdamW':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay
        )
    elif cfg.optimizer == 'SGD':
        optimizer = optim.SGD(
            model.parameters(),
            lr=cfg.lr,
            momentum=0.9,
            weight_decay=cfg.weight_decay
        )
    else:
        raise NotImplementedError(f"Optimizer {cfg.optimizer} not implemented")
        
    return optimizer


def get_scheduler(optimizer, cfg, steps_per_epoch=None):
    if cfg.scheduler == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.T_max,
            eta_min=cfg.min_lr
        )
    elif cfg.scheduler == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=2,
            min_lr=cfg.min_lr,
        )
    elif cfg.scheduler == 'StepLR':
        scheduler = lr_scheduler.StepLR(
            optimizer,
            step_size=cfg.epochs // 3,
            gamma=0.5
        )
    elif cfg.scheduler == 'OneCycleLR':
        scheduler = lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=cfg.lr,
            epochs=cfg.epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=cfg.pct_start,
        )
    else:
        scheduler = None
        
    return scheduler

class FocalLossBCE(torch.nn.Module):
    def __init__(
            self,
            alpha: float,
            gamma: float,
            reduction: str,
            bce_weight: float,
            focal_weight: float,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce = torch.nn.BCEWithLogitsLoss(reduction=reduction)
        self.bce_weight = bce_weight
        self.focal_weight = focal_weight

    def forward(self, logits, targets):
        focall_loss = torchvision.ops.focal_loss.sigmoid_focal_loss(
            inputs=logits,
            targets=targets,
            alpha=self.alpha,
            gamma=self.gamma,
            reduction=self.reduction,
        )
        bce_loss = self.bce(logits, targets)
        return self.bce_weight * bce_loss + self.focal_weight * focall_loss


def get_criterion(cfg):
 
    if cfg.criterion == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss()
    elif cfg.criterion == 'FocalLossBCE':
        criterion = FocalLossBCE(
            alpha=cfg.flbce_alpha, 
            gamma=cfg.flbce_gamma, 
            reduction=cfg.flbce_reduction,
            bce_weight=cfg.flbce_bce_weight,
            focal_weight=cfg.flbce_focal_weight
        )
    else:
        raise NotImplementedError(f"Criterion {cfg.criterion} not implemented")
        
    return criterion


def train_one_epoch(model, loader, optimizer, criterion, device, cur_epoch, scheduler=None):
    model.train()
    losses = []
    all_targets = []
    all_outputs = []
    
    pbar = tqdm(enumerate(loader), total=len(loader), desc="Training")
    
    global_step = cur_epoch * len(loader)
    
    for step, batch in pbar:
        current_step = global_step + step
        if isinstance(batch['melspec'], list):
            batch_outputs = []
            batch_losses = []
            
            for i in range(len(batch['melspec'])):
                inputs = batch['melspec'][i].unsqueeze(0).to(device)
                target = batch['target'][i].unsqueeze(0).to(device)
                
                optimizer.zero_grad()
                output = model(inputs)
                loss = criterion(output, target)
                loss.backward()
                
                batch_outputs.append(output.detach().cpu())
                batch_losses.append(loss.item())
            
            optimizer.step()
            outputs = torch.cat(batch_outputs, dim=0).numpy()
            loss = np.mean(batch_losses)
            targets = batch['target'].numpy()
            
        else:
            inputs = batch['melspec'].to(device)
            targets = batch['target'].to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            if isinstance(outputs, tuple):
                outputs, loss = outputs  
            else:
                loss = criterion(outputs, targets)
                
            loss.backward()
            optimizer.step()
            
            outputs = outputs.detach().cpu().numpy()
            targets = targets.detach().cpu().numpy()
        
        if scheduler is not None and isinstance(scheduler, lr_scheduler.OneCycleLR):
            scheduler.step()
            
        all_outputs.append(outputs)
        all_targets.append(targets)
        losses.append(loss if isinstance(loss, float) else loss.item())
        
        train_loss = np.mean(losses[-cfg.batch_size:]) if losses else 0
        train_auc = calculate_auc(targets, outputs)
        lr = optimizer.param_groups[0]['lr']
        
        wandb.log({
            'train/step': current_step,
            'train/loss': train_loss, 
            'train/auc': train_auc, 
            'train/lr': lr
        })
        
        pbar.set_postfix({
            'loss': train_loss,
            'lr': lr
        })

    all_outputs = np.concatenate(all_outputs)
    all_targets = np.concatenate(all_targets)
    auc = calculate_auc(all_targets, all_outputs)
    avg_loss = np.mean(losses)
    
    return avg_loss, auc


def validate(model, loader, criterion, device, cur_epoch):
    model.eval()
    losses = []
    all_targets = []
    all_outputs = []
    
    global_step = cur_epoch * len(loader)
    
    with torch.inference_mode():
        pbar = tqdm(enumerate(loader), total=len(loader), desc="Validation")
        
        for step, batch in pbar:
            current_step = global_step + step
            if isinstance(batch['melspec'], list):
                batch_outputs = []
                batch_losses = []
                
                for i in range(len(batch['melspec'])):
                    inputs = batch['melspec'][i].unsqueeze(0).to(device)
                    target = batch['target'][i].unsqueeze(0).to(device)
                    
                    output = model(inputs)
                    loss = criterion(output, target)
                    
                    batch_outputs.append(output.detach().cpu())
                    batch_losses.append(loss.item())
                
                outputs = torch.cat(batch_outputs, dim=0).numpy()
                loss = np.mean(batch_losses)
                targets = batch['target'].numpy()
                
            else:
                inputs = batch['melspec'].to(device)
                targets = batch['target'].to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                outputs = outputs.detach().cpu().numpy()
                targets = targets.detach().cpu().numpy()
            
            all_outputs.append(outputs)
            all_targets.append(targets)
            losses.append(loss if isinstance(loss, float) else loss.item())
            val_loss = np.mean(losses[-cfg.batch_size:]) if losses else 0
            val_auc = calculate_auc(targets, outputs)
            
            wandb.log({
                'val/step': current_step,
                'val/loss': val_loss, 
                'val/auc': val_auc
            })

    all_outputs = np.concatenate(all_outputs)
    all_targets = np.concatenate(all_targets)
    
    auc = calculate_auc(all_targets, all_outputs)
    avg_loss = np.mean(losses)
    
    return avg_loss, auc


def run_training(cfg, spectrograms):
    df = pd.read_csv(cfg.train_csv)
    taxonomy_df = pd.read_csv(cfg.taxonomy_csv)
    species_ids = taxonomy_df['primary_label'].tolist()
    cfg.num_classes = len(species_ids)
    
    if cfg.debug:
        cfg.update_debug_settings()

    skf = StratifiedKFold(n_splits=cfg.n_fold, shuffle=True, random_state=cfg.seed)
    
    best_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['primary_label'])):
        if fold not in cfg.selected_folds:
            continue

        wandb.init(
            project='birdclefplus_2025',
            name=f'{cfg.exp_name}_f{fold}',
            config=cfg.to_dict()
        )
        wandb.define_metric("train/*", step_metric="train/step")
        wandb.define_metric("val/*", step_metric="val/step")
        wandb.define_metric("epoch/*", step_metric="epoch/step")

        print(f'\n{"="*30} Fold {fold} {"="*30}')
        
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)
        
        print(f'Training set: {len(train_df)} samples')
        print(f'Validation set: {len(val_df)} samples')
        
        train_dataset = BirdCLEFDatasetFromNPY(train_df, cfg, spectrograms=spectrograms, mode='train')
        val_dataset = BirdCLEFDatasetFromNPY(val_df, cfg, spectrograms=spectrograms, mode='valid')
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=cfg.batch_size, 
            shuffle=True, 
            num_workers=cfg.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=cfg.batch_size, 
            shuffle=False, 
            num_workers=cfg.num_workers,
            pin_memory=True,
            collate_fn=collate_fn
        )
        
        model = BirdCLEFModel(cfg).to(cfg.device)
        optimizer = get_optimizer(model, cfg)
        criterion = get_criterion(cfg)
        
        scheduler = get_scheduler(optimizer, cfg, len(train_loader))
        
        best_auc = 0
        best_epoch = 0
        
        for epoch in range(cfg.epochs):
            print(f"\nEpoch {epoch+1}/{cfg.epochs}")
            
            train_loss, train_auc = train_one_epoch(
                model, 
                train_loader, 
                optimizer, 
                criterion, 
                cfg.device,
                epoch,
                scheduler if isinstance(scheduler, lr_scheduler.OneCycleLR) else None
            )
            val_loss, val_auc = validate(model, val_loader, criterion, cfg.device, epoch)
            wandb.log({
                'epoch/step': epoch,
                'epoch/train_loss': train_loss, 
                'epoch/train_auc': train_auc, 
                'epoch/val_loss': val_loss, 
                'epoch/val_auc': val_auc
            })

            if scheduler is not None and not isinstance(scheduler, lr_scheduler.OneCycleLR):
                if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()

            print(f"Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}")
            
            if val_auc > best_auc:
                best_auc = val_auc
                best_epoch = epoch + 1
                print(f"New best AUC: {best_auc:.4f} at epoch {best_epoch}")

                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                    'epoch': epoch,
                    'val_auc': val_auc,
                    'train_auc': train_auc,
                    'cfg': cfg.to_dict()
                }, join(cfg.OUTPUT_DIR, cfg.exp_name, f"model_{epoch:02}_f{fold}-{best_auc:.6f}.pth"))
        
        best_scores.append(best_auc)
        print(f"\nBest AUC for fold {fold}: {best_auc:.4f} at epoch {best_epoch}")
        
        wandb.finish()
        # Clear memory
        del model, optimizer, scheduler, train_loader, val_loader
        torch.cuda.empty_cache()
        gc.collect()

    print("\n" + "="*60)
    print("Cross-Validation Results:")
    for fold, score in enumerate(best_scores):
        print(f"Fold {cfg.selected_folds[fold]}: {score:.4f}")
    print(f"Mean AUC: {np.mean(best_scores):.4f}")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BirdCLEF model")
    parser.add_argument('-c', '--config', type=str, default='configs/debug.yaml', help='Path to config file')
    args = parser.parse_args()
    cfg = Config(args.config)
    now_str = datetime.datetime.now(datetime.timezone.utc).strftime("%d-%m_%H-%M-%S")
    cfg.exp_name = f"{splitext(basename(args.config))[0]}_{now_str}"

    set_seed(cfg.seed)
    load_dotenv()
    wandb.login()

    os.makedirs(join(cfg.OUTPUT_DIR, cfg.exp_name), exist_ok=True)

    if cfg.precompute_audio:
        print("Precomputing mel spectrograms...")
        spectrograms = process_data(cfg)
    else:
        spectrograms = None
    print("\nStarting training...")
    run_training(cfg, spectrograms)
    print("\nTraining complete!")
