import pandas as pd
from tqdm import tqdm
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'SimHei'
import joblib
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import r2_score, mean_absolute_error
import torch.nn.init as init
from tensorboardX import SummaryWriter
from resnet_bilstm import UltraTermBiLSTM  # Import model
import os
import glob
from datetime import datetime

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Model evaluation function
def evaluate_model(model, dataloader, criterion):
    model.eval()
    total_loss = 0.0
    running_r2 = 0.0
    running_mae = 0.0
    running_acc = 0.0
    running_max_y = 0.0
    running_acc15 = 0.0
    running_mae15 = 0.0
    with torch.no_grad():
        for i, data in enumerate(dataloader, 0):
            real_input, fcst_input, labels = data
            outputs = model(real_input, fcst_input)
            loss = criterion(outputs, labels)
            y_true = labels.cpu().numpy()
            y_pred = outputs.detach().cpu().numpy()
            r2 = r2_score(y_true=y_true, y_pred=y_pred)
            acc = acc_solar(y_true[:, 0], y_pred[:, 0], MW=200)
            acc_15 = acc_solar(y_true[:, 15], y_pred[:, 15], MW=200)
            mae = mean_absolute_error(y_true[:, 0], y_pred[:, 0])
            mae_15 = mean_absolute_error(y_true[:, 15], y_pred[:, 15])
            running_r2 += r2
            running_mae += mae
            running_acc += acc
            running_acc15 += acc_15
            running_mae15 += mae_15
            running_max_y += np.max(y_pred)
            total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    avg_r2 = running_r2 / len(dataloader)
    avg_mae = running_mae / len(dataloader)
    avg_acc = running_acc / len(dataloader)
    max_y = running_max_y / len(dataloader)
    avg_acc15 = running_acc15 / len(dataloader)
    avg_mae15 = running_mae15 / len(dataloader)
    model.train()
    print(f'MAE point 1: {avg_mae:.2f}, MAE point 16: {avg_mae15:.2f}, Acc point 1: {avg_acc:.2f}, Acc point 16: {avg_acc15:.2f}')
    return avg_loss, avg_r2, avg_mae, avg_acc, max_y

# Function to test model and save results
def test_and_save_results(model, test_data_list, output_path):
    model.eval()
    device = next(model.parameters()).device
    
    with torch.no_grad():
        # Merge test data from all months
        all_test_real = []
        all_test_fcst = []
        all_test_label = []
        all_time_info = []
        
        for test_data in test_data_list:
            if len(test_data['test_real']) > 0:
                all_test_real.append(test_data['test_real'])
                all_test_fcst.append(test_data['test_fcst'])
                all_test_label.append(test_data['test_label'])
                all_time_info.extend(test_data['time_info'])
        
        # Return if no test data available
        if not all_test_real:
            print("Warning: No test data available")
            return
            
        # Concatenate all test data
        test_real = torch.cat(all_test_real, dim=0).to(device)
        test_fcst = torch.cat(all_test_fcst, dim=0).to(device)
        test_label = torch.cat(all_test_label, dim=0)
        
        # Perform prediction
        outputs = model(test_real, test_fcst)
        outputs_np = outputs.cpu().numpy()
        test_label_np = test_label.numpy()
        test_fcst_np = test_fcst.cpu().numpy()
        
        # Create results DataFrame
        result = pd.DataFrame()
        
        # For each forecast time point
        for i in range(outputs_np.shape[1]):
            # Calculate hours and minutes
            hours = i // 4  # Integer division gives hours
            minutes = (i % 4) * 15  # Remainder * 15 gives minutes
            
            # Get corresponding forecast times
            forecast_times = [info['forecast_times'][i] for info in all_time_info]
            
            temp_df = pd.DataFrame({
                'DATETIME': forecast_times,  # Actual forecast times
                f'errataPower_{hours}h{minutes}min': outputs_np[:, i],     # Corrected result for each time point
                f'truePower_{hours}h{minutes}min': test_label_np[:, i],    # True value for each time point
                f'Fcst_Power_{hours}h{minutes}min': test_fcst_np[:, -1, i]  # Forecast power for each time point
            })
            
            # Assign to result DataFrame
            if i == 0:
                result = temp_df
            else:
                result[f'errataPower_{hours}h{minutes}min'] = temp_df[f'errataPower_{hours}h{minutes}min']
                result[f'truePower_{hours}h{minutes}min'] = temp_df[f'truePower_{hours}h{minutes}min']
                result[f'Fcst_Power_{hours}h{minutes}min'] = temp_df[f'Fcst_Power_{hours}h{minutes}min']
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save results
        result.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"Merged test results saved to {output_path}")
        
        # Calculate and print test set evaluation metrics
        mae = mean_absolute_error(test_label_np.flatten(), outputs_np.flatten())
        r2 = r2_score(test_label_np.flatten(), outputs_np.flatten())
        print(f"Merged test set evaluation: MAE: {mae:.2f}, R2: {r2:.4f}")

# Weighted MAE loss function
def weighted_mae_loss(outputs, labels):
    weights = torch.linspace(1, 1.5, 16).to(outputs.device)  # Higher weight for later time steps
    return torch.mean(weights * torch.abs(outputs - labels))

# Function to merge datasets
def merge_datasets(data_files):
    """
    Merge datasets from all months
    """
    merged_data = {
        'train_real': [],
        'train_fcst': [],
        'train_label': [],
        'val_real': [],
        'val_fcst': [],
        'val_label': [],
        'test_real': [],
        'test_fcst': [],
        'test_label': [],
        'time_info': [],
        'selected_features': None,
        'config': None
    }
    
    for data_file in data_files:
        month_data = joblib.load(data_file)
        
        # Merge training data
        if len(month_data['train_real']) > 0:
            merged_data['train_real'].append(month_data['train_real'])
            merged_data['train_fcst'].append(month_data['train_fcst'])
            merged_data['train_label'].append(month_data['train_label'])
        
        # Merge validation data
        if len(month_data['val_real']) > 0:
            merged_data['val_real'].append(month_data['val_real'])
            merged_data['val_fcst'].append(month_data['val_fcst'])
            merged_data['val_label'].append(month_data['val_label'])
        
        # Merge test data
        if len(month_data['test_real']) > 0:
            merged_data['test_real'].append(month_data['test_real'])
            merged_data['test_fcst'].append(month_data['test_fcst'])
            merged_data['test_label'].append(month_data['test_label'])
            merged_data['time_info'].extend(month_data['time_info'])
        
        # Save feature and configuration information
        if merged_data['selected_features'] is None:
            merged_data['selected_features'] = month_data['selected_features']
            merged_data['config'] = month_data['config']
    
    # Convert all data to tensors
    for key in ['train_real', 'train_fcst', 'train_label', 
                'val_real', 'val_fcst', 'val_label',
                'test_real', 'test_fcst', 'test_label']:
        if merged_data[key]:
            merged_data[key] = torch.cat(merged_data[key], dim=0)
        else:
            merged_data[key] = torch.tensor([])
    
    return merged_data

# Function to train BiLSTM model
def train_bilstm_model(merged_data, output_dir, num_epochs=100, patience=5):
    # Device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create log and model save directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/logs", exist_ok=True)
    os.makedirs(f"{output_dir}/models", exist_ok=True)
    os.makedirs(f"{output_dir}/results", exist_ok=True)
    
    # Initialize TensorBoard
    log_dir = f"{output_dir}/logs/bilstm_model"
    writer = SummaryWriter(log_dir, flush_secs=10)
    
    # Check if training set is empty
    if merged_data['train_real'].shape[0] == 0:
        print("Warning: Merged training set is empty, cannot train")
        return None
    
    # Create data loaders
    train_dataset = TensorDataset(merged_data['train_real'].to(device), 
                                 merged_data['train_fcst'].to(device), 
                                 merged_data['train_label'].to(device))
    
    val_dataset = TensorDataset(merged_data['val_real'].to(device), 
                               merged_data['val_fcst'].to(device), 
                               merged_data['val_label'].to(device))
    
    batch_size = 32
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    # Create validation data loader if validation set is not empty
    if merged_data['val_real'].shape[0] > 0:
        dataloader_val = DataLoader(val_dataset, batch_size=128, shuffle=False)
    else:
        dataloader_val = None
    
    # Model initialization - using BiLSTM model
    real_in = merged_data['train_real'].shape[1]
    fcst_in = merged_data['train_fcst'].shape[1]
    model = UltraTermBiLSTM(real_in=real_in, forecast_in=fcst_in, real_out=32, forecast_out=32)
    model = model.to(device)
    
    # Parameter initialization
    for param in model.parameters():
        if len(param.size()) > 1:
            init.xavier_uniform_(param)
    
    # Optimizer and learning rate scheduler
    criterion = nn.L1Loss()
    #optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    optimizer = optim.SGD(model.parameters(), lr=2e-4, momentum=0.9)
    #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # Training loop
    min_epochs_to_trigger = 5
    best_loss = np.inf
    epochs_no_improve = 0
    train_list = []
    val_list = []
    best_val_loss = np.inf
    model_path = f"{output_dir}/models/best_bilstm_model.pth"
    model.train()
    
    print("Starting BiLSTM model training")
    print(f"Training set size: {merged_data['train_real'].shape[0]}")
    print(f"Validation set size: {merged_data['val_real'].shape[0]}")
    print(f"Test set size: {merged_data['test_real'].shape[0]}")
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        running_r2 = 0.0
        running_mae = 0.0
        
        # Training
        for i, data in enumerate(dataloader, 0):
            real_input, fcst_input, labels = data
            optimizer.zero_grad()
            outputs = model(real_input, fcst_input)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            y_true = labels.cpu().numpy()
            y_pred = outputs.detach().cpu().numpy()
            r2 = r2_score(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            running_r2 += r2
            running_mae += mae
        
        scheduler.step()
        
        # Calculate average training metrics
        avg_loss = running_loss / len(dataloader)
        avg_r2 = running_r2 / len(dataloader)
        avg_mae = running_mae / len(dataloader)
        
        # Validation
        if dataloader_val is not None:
            val_loss, val_r2, val_mae, val_acc, max_y = evaluate_model(model, dataloader_val, criterion)
            val_metrics_msg = f", Val Loss: {val_loss:.4f}, Val R2: {val_r2:.2f}, Val MAE: {val_mae:.2f}, Val Acc: {val_acc:.2f}"
        else:
            val_loss = avg_loss  # Use training loss if no validation set
            val_r2 = avg_r2
            val_mae = avg_mae
            val_metrics_msg = ", no validation set"
        
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, R2: {avg_r2:.4f}, MAE: {avg_mae:.2f}{val_metrics_msg}")
        
        train_list.append(avg_loss)
        val_list.append(val_loss)
        writer.add_scalars("loss", {'train loss': avg_loss, 'val loss': val_loss}, epoch)
        writer.add_scalars("R2", {'train R2': avg_r2, 'val R2': val_r2}, epoch)
        
        # Save best model
        if epoch >= min_epochs_to_trigger:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), model_path)
                print(f"Best model saved (Val Loss: {val_loss:.6f})")
                    
                # Reset early stopping counter
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                
            # Early stopping
            if epochs_no_improve >= patience and epoch >= 3:
                print(f"Early stopping triggered, no improvement for {patience} epochs")
                break
    
    writer.close()
    print("BiLSTM model training completed")
    
    # Load best model
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print(f"Loaded best model: {model_path}")
    
    return model

# Main function: merge training and test BiLSTM model
def merged_train_test_bilstm(data_dir, output_dir):
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all data files
    data_files = sorted(glob.glob(f"{data_dir}/wailuo*_cleaned*.job"))
    
    if not data_files:
        print(f"Error: No data files found in {data_dir}")
        return
        
    print(f"Found {len(data_files)} monthly data files")
    
    # Merge data from all months
    print("\nMerging data from all months...")
    merged_data = merge_datasets(data_files)
    
    # Train BiLSTM model
    model = train_bilstm_model(merged_data, output_dir, patience=10)
    
    if model is not None:
        # Load test data from all months
        test_data_list = []
        for data_file in data_files:
            month_data = joblib.load(data_file)
            test_data_list.append(month_data)
        
        # Merge test results and save
        test_output_path = f"{output_dir}/results/bilstm_test_results.csv"
        test_and_save_results(model, test_data_list, test_output_path)
    
    print("\nAll monthly data processing completed")

# Main program
if __name__ == "__main__":

    print(f"Starting training ")
    data_dir = f"./monthly_split_windAVE_windMAX_windMIN_66"  # Monthly split dataset directory
    output_dir = f"./output"  # Output directory
    
    merged_train_test_bilstm(data_dir, output_dir)
