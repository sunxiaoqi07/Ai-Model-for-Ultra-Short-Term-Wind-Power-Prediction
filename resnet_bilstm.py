import torch 
from torch import nn 

from torchvision.transforms import transforms
import torch.nn.functional as F 



class FeedForward(nn.Module):  
    def __init__(self, d_model, d_ff, dropout=0.1):  
        super(FeedForward, self).__init__()  
        # Two linear layers: first maps d_model to d_ff, second maps d_ff back to d_model
        self.linear1 = nn.Linear(d_model, d_ff)  
        self.dropout = nn.Dropout(dropout)  
        self.linear2 = nn.Linear(d_ff, d_model)  
    def forward(self, x):  
        # First linear layer
        x = self.linear1(x)  
        # Apply ELU activation function
        x = F.elu(x)  
        # Apply dropout
        x = self.dropout(x)  
        # Second linear layer
        x = self.linear2(x)  
        return x  

class PositionalEncoding(nn.Module):
    """Positional Encoding"""
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # Create a sufficiently long P matrix
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)



class UltraTermBiLSTM(nn.Module):
    """Ultra-short-term wind power forecasting - BiLSTM version"""
    def __init__(self, real_in, forecast_in, real_out, forecast_out) -> None:
        super(UltraTermBiLSTM,self).__init__()
        
        self.real_inchannel = real_in  # Real observation input channels
        self.forecast_inchannel = forecast_in  # Forecast input channels
        self.real_outchannel = real_out  # Real observation output channels
        self.forecast_outchannel = forecast_out  # Forecast output channels
        
        self.fc_nums = 512  # Unified dimension for BiLSTM hidden layers
        
        self.real_encoder = ResNet1D(self.real_inchannel, self.real_outchannel)  # Extract features from real observation data
        self.forecast_encoder = ResNet1D(self.forecast_inchannel, self.forecast_outchannel)  # Extract features from forecast data

        self.blance_real = nn.Linear(self.real_outchannel, self.fc_nums)  # Unify dimensions
        self.blance_forecast = nn.Linear(self.forecast_outchannel, self.fc_nums)  # Unify dimensions
        
        ## Add positional encoding
        self.position_encoding_real = PositionalEncoding(self.fc_nums, dropout=0.0, max_len=96)
        self.position_encoding_forecast = PositionalEncoding(self.fc_nums, dropout=0.0, max_len=96)
        
        # Replace Transformer layers with BiLSTM
        self.real_bilstm = nn.LSTM(
            input_size=self.fc_nums,
            hidden_size=self.fc_nums//2,  # Bidirectional LSTM, half hidden size, concatenated forward and backward to match dimension
            num_layers=2,
            batch_first=True,
            dropout=0.2,
            bidirectional=True
        )
        self.real_layer_normal = nn.LayerNorm(self.fc_nums)
        self.real_dropout = nn.Dropout(0.2)
        self.real_ffn = FeedForward(self.fc_nums, self.fc_nums, dropout=0.2)
        self.real_feednorm = nn.LayerNorm(self.fc_nums)
        
        # Forecast BiLSTM
        self.fcst_bilstm = nn.LSTM(
            input_size=self.fc_nums,
            hidden_size=self.fc_nums//2,
            num_layers=2,
            batch_first=True,
            dropout=0.4,
            bidirectional=True
        )
        self.fcst_layer_norm = nn.LayerNorm(self.fc_nums)
        self.fcst_dropout = nn.Dropout(0.4)
        
        # Cross BiLSTM layer - used to fuse real observation and forecast information
        self.cross_bilstm = nn.LSTM(
            input_size=self.fc_nums*2,  # Input is concatenation of real observation and forecast features
            hidden_size=self.fc_nums//2,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
            bidirectional=True
        )
        self.cross_dropout = nn.Dropout(0.2)
        self.cross_layer_norm = nn.LayerNorm(self.fc_nums)
        self.cross_ffn = FeedForward(self.fc_nums, self.fc_nums, dropout=0.2)
        self.cross_feed_norm = nn.LayerNorm(self.fc_nums)

        self.fc_out = nn.Linear(self.fc_nums, 1)  # Final output layer
        
    
    def forward(self, real_seq, forecast_seq):
        ## Pass real_seq and forecast_seq through encoders respectively
        # Real observation encoder
        real_out = self.real_encoder(real_seq)
        
        # Forecast encoder
        forecast_out = self.forecast_encoder(forecast_seq)
        
        B_r, C_r, L_r = real_out.shape
        B_f, C_f, L_f = forecast_out.shape
        
        # Permute dimensions
        real_out = real_out.permute(0, 2, 1)  # [batch, seq_len, channel]
        forecast_out = forecast_out.permute(0, 2, 1)
        
        # Unify channel dimensions
        real_out = self.blance_real(real_out)
        forecast_out = self.blance_forecast(forecast_out)
        
        # Positional encoding
        real_out = self.position_encoding_real(real_out)
        forecast_out = self.position_encoding_forecast(forecast_out)
        
        # Process real observation data with BiLSTM
        real_lstm_out, _ = self.real_bilstm(real_out)
        real_lstm_out = self.real_dropout(real_lstm_out)
        real_lstm_out = self.real_layer_normal(real_out + real_lstm_out)  # Residual connection
        
        real_feed_out = self.real_ffn(real_lstm_out)
        real_lstm_out = self.real_feednorm(real_feed_out + real_lstm_out)  # Residual connection
        
        # Process forecast data with BiLSTM
        fcst_lstm_out, _ = self.fcst_bilstm(forecast_out)
        fcst_lstm_out = self.fcst_dropout(fcst_lstm_out)
        fcst_lstm_out = self.fcst_layer_norm(forecast_out + fcst_lstm_out)  # Residual connection
        
        # Cross information fusion
        # Concatenate forecast and real observation features
        cross_input = torch.cat([fcst_lstm_out, real_lstm_out], dim=-1)
        
        # Process fused features with cross BiLSTM
        cross_lstm_out, _ = self.cross_bilstm(cross_input)
        cross_lstm_out = self.cross_dropout(cross_lstm_out)
        
        # Take cross BiLSTM output, apply residual connection and layer normalization
        # Since input and output dimensions differ, we only use the BiLSTM output
        cross_lstm_out = self.cross_layer_norm(cross_lstm_out)
        
        # Further process with feedforward network
        cross_feed_out = self.cross_ffn(cross_lstm_out)
        cross_lstm_out = self.cross_feed_norm(cross_feed_out + cross_lstm_out)
        
        # Final output
        out = self.fc_out(cross_lstm_out)
        out = out.squeeze(dim=2)  # Squeeze dimension
        
        return out
    
    




class BasicBlock1D(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1):
        super(BasicBlock1D, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channel)
        self.relu = nn.ReLU(inplace=False)
        
        self.conv2 = nn.Conv1d(out_channel, out_channel, kernel_size=kernel_size, stride=1, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channel)
        
        self.downsample = None
        if stride != 1 or in_channel != out_channel:
            # Use 1x1 convolution to match channel dimensions
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channel, out_channel, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channel)
            )
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out = out + identity
        out = self.relu(out)
        
        return out

class ResNet1D(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ResNet1D, self).__init__()
        
        # Initial convolutional layer
        self.conv1 = nn.Conv1d(in_channel, 16, kernel_size=7, stride=1, padding=3)
        self.bn1 = nn.BatchNorm1d(16)
        self.relu = nn.ReLU(inplace=False)
        
        # ResNet residual blocks (multiple BasicBlocks can be stacked)
        self.resblock1 = BasicBlock1D(16, out_channel)
        #self.resblock2 = BasicBlock1D(32, out_channel)

    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.resblock1(x)
        #x = self.resblock2(x)

        
        return x
