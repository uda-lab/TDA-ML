import argparse 
import torch 
import os 
import csv 
import sys 

from src.utils import load_config
from src.data_loader import NoisyMNISTDataset, create_data_loader
from src.models import AnisotropicOutlierClassifier
from src.trainer import Trainer

def main(config_name=None, config=None, trial=None, config_overrides=None):
    if config is None:
        if config_name is None:
            raise ValueError("Either config_name or config must be provided")
        config = load_config(config_name)
        
    if config_overrides:
        config.update(config_overrides)
    
    print(f"Loaded config: {config['meta'].get('config_id', 'unknown')}")

    if config.get('device') and config['device'] != 'auto':
        device = torch.device(config['device'])
    elif torch.cuda.is_available():
        device = torch.device('cuda') 
    elif torch.backends.mps.is_available():
        device = torch.device('mps') 
    else:
        device = torch.device('cpu')

    import datetime
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    config_id = config['meta'].get('config_id', 'unknown')
    
    # --- Directory Setup ---
    if 'outputs' in config:
        base_dir = config['outputs'].get('base_dir', 'outputs')
        run_dir_name = f"{config_id}_{timestamp}"
        run_dir = os.path.join(base_dir, run_dir_name)
        
        # Consistent directory names across the project
        config['outputs']['log_dir'] = os.path.join(run_dir, 'logs')
        config['outputs']['image_dir'] = os.path.join(run_dir, 'images')
        
        # Note: Trainer also calls makedirs to ensure robustness
        os.makedirs(config['outputs']['log_dir'], exist_ok=True)
        os.makedirs(config['outputs']['image_dir'], exist_ok=True)
        
        print(f"Project structure created at: {run_dir}")

    data_cfg = config['data']
    seed = data_cfg.get('seed', 42)
    train_size = data_cfg.get('train_size', 4500)
    val_size = data_cfg.get('val_size', 500)
    test_size = data_cfg.get('test_size', 1000)
    
    generator = torch.Generator().manual_seed(seed)
    full_train_indices = torch.randperm(60000, generator=generator)[:train_size + val_size]
    
    train_indices = full_train_indices[:train_size]
    val_indices = full_train_indices[train_size:]
    
    optimal_workers = 8 
    use_pin_memory = True if device.type != 'cpu' else False

    train_dataset = NoisyMNISTDataset(
        root='./data', train=True, max_points=data_cfg['max_points'],
        num_outliers=data_cfg['num_outliers'], indices=train_indices,
        deterministic=True, noise_seed=seed
    )
    
    data_loader = create_data_loader(
        train_dataset, 
        batch_size=data_cfg['batch_size'], 
        shuffle=True, 
        num_workers=optimal_workers, 
        pin_memory=use_pin_memory
    )
    
    val_dataset = NoisyMNISTDataset(
        root='./data', train=True, max_points=data_cfg['max_points'],
        num_outliers=data_cfg['num_outliers'], indices=val_indices,
        deterministic=True, noise_seed=seed
    )
    
    val_loader = create_data_loader(
        val_dataset, 
        batch_size=data_cfg['batch_size'], 
        shuffle=False, 
        num_workers=optimal_workers, 
        pin_memory=use_pin_memory
    )

    test_indices = torch.randperm(10000, generator=generator)[:test_size]
    test_dataset = NoisyMNISTDataset(
        root='./data', train=False, max_points=data_cfg['max_points'],
        num_outliers=data_cfg['num_outliers'], indices=test_indices,
        deterministic=True, noise_seed=seed
    )
    
    test_loader = create_data_loader(
        test_dataset, 
        batch_size=data_cfg['batch_size'], 
        shuffle=False, 
        num_workers=optimal_workers, 
        pin_memory=use_pin_memory
    )

    model = AnisotropicOutlierClassifier() 
    model.to(device) 

    trainer = Trainer(model, config, device=device) 

    init_checkpoint = config.get('init_checkpoint')
    if init_checkpoint and os.path.exists(init_checkpoint):
        print(f"Loading initial weights from {init_checkpoint}")
        checkpoint = torch.load(init_checkpoint, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
    elif init_checkpoint:
        print(f"Warning: Initial checkpoint not found at {init_checkpoint}")

    log_dir = config['outputs']['log_dir']
    metrics_path = os.path.join(log_dir, 'metrics.csv')
    with open(metrics_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'train_class_loss', 'train_topo_loss', 'train_aniso_loss', 'train_size_loss', 'val_loss', 'val_recall', 'val_mcc', 'val_aniso', 'val_size'])

    # --- Training Loop ---
    epochs = config['training']['epochs']
    save_every = config['outputs'].get('save_every', 10)
    best_val_mcc = -1.0 
    
    for epoch in range(1, epochs + 1):
        # res returns (avg_loss, class_loss, topo_loss, aniso_loss, size_loss, ...)
        res = trainer.train_epoch(data_loader, epoch) 
        val_res = trainer.validate(val_loader)
        
        val_mcc = val_res[4] # MCC is at index 4
        print(f"Epoch {epoch}: Val MCC={val_mcc:.4f}, Aniso={val_res[5]:.4f}") 
        
        # Save best model logic
        if val_mcc > best_val_mcc:
            best_val_mcc = val_mcc
            best_model_path = os.path.join(run_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_mcc': val_mcc,
            }, best_model_path)
            print(f"Saved best model (MCC: {val_mcc:.4f}) to {best_model_path}")

        if epoch % save_every == 0:
            checkpoint_path = os.path.join(run_dir, f'checkpoint_epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_mcc': val_mcc,
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

        with open(metrics_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, res[0], res[1], res[2], res[3], res[4], val_res[0], val_res[1], val_res[4], val_res[5], val_res[6]])
            
    final_model_path = os.path.join(run_dir, 'final_model.pth')
    torch.save(model.state_dict(), final_model_path)
    print(f"Saved final model to {final_model_path}")
    
    return val_res[3]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='dev', help='Config path')
    parser.add_argument('--init_checkpoint', type=str, default=None, help='Path to initial checkpoint')
    args = parser.parse_args()
    
    config_overrides = {}
    if args.init_checkpoint:
        config_overrides['init_checkpoint'] = args.init_checkpoint
        
    main(config_name=args.config, config_overrides=config_overrides) 
