import torch 
from torch import nn 

from torchvision.transforms import transforms
import torch.nn.functional as F 



class FeedForward(nn.Module):  
    def __init__(self, d_model, d_ff, dropout=0.1):  
        super(FeedForward, self).__init__()  
        # 两个线性层：第一层将输入维度d_model映射到d_ff，第二层将d_ff映射回d_model  
        self.linear1 = nn.Linear(d_model, d_ff)  
        self.dropout = nn.Dropout(dropout)  
        self.linear2 = nn.Linear(d_ff, d_model)  
    def forward(self, x):  
        # 第一个线性层  
        x = self.linear1(x)  
        # 应用ReLU激活函数  
        x = F.elu(x)  
        # 应用dropout  
        x = self.dropout(x)  
        # 第二个线性层  
        x = self.linear2(x)  
        return x  

class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # 创建一个足够长的P,广播机制生成的X.shape为（ max_len, arange(0, num_hiddens, 2) )
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
    """风电超短期预测 - BiLSTM版本"""
    def __init__(self, real_in, forecast_in, real_out, forecast_out) -> None:
        super(UltraTermBiLSTM,self).__init__()
        
        self.real_inchannel=real_in # 实况输入通道
        self.forecast_inchannel=forecast_in # 预报输入通道数
        self.real_outchannel=real_out # 实况输出通道数
        self.forecast_outchannel=forecast_out # 预报输出通道数
        
        self.fc_nums=512  # BiLSTM隐藏层的统一维度
        
        self.real_encoder=ResNet1D(self.real_inchannel, self.real_outchannel) # 处理实况数据，提取特征
        self.forecast_encoder=ResNet1D(self.forecast_inchannel, self.forecast_outchannel) # 处理预报数据，提取特征

        self.blance_real=nn.Linear(self.real_outchannel,self.fc_nums)  # 统一维度
        self.blance_forecast=nn.Linear(self.forecast_outchannel,self.fc_nums) # 统一维度
        
        ## 添加位置编码
        self.position_encoding_real = PositionalEncoding(self.fc_nums,dropout=0.0,max_len=96)
        self.position_encoding_forecast = PositionalEncoding(self.fc_nums,dropout=0.0,max_len=96)
        
        # 替换Transformer层为BiLSTM
        self.real_bilstm = nn.LSTM(
            input_size=self.fc_nums,
            hidden_size=self.fc_nums//2,  # 双向LSTM，隐藏层为一半大小，正向反向拼接后维度一致
            num_layers=2,
            batch_first=True,
            dropout=0.2,
            bidirectional=True
        )
        self.real_layer_normal = nn.LayerNorm(self.fc_nums)
        self.real_dropout = nn.Dropout(0.2)
        self.real_ffn = FeedForward(self.fc_nums, self.fc_nums, dropout=0.2)
        self.real_feednorm = nn.LayerNorm(self.fc_nums)
        
        # 预报BiLSTM
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
        
        # 交叉BiLSTM层 - 用于融合实况和预报信息
        self.cross_bilstm = nn.LSTM(
            input_size=self.fc_nums*2,  # 输入是实况和预报的拼接
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

        self.fc_out = nn.Linear(self.fc_nums, 1)  # 最终输出层
        
    
    def forward(self, real_seq, forecast_seq):
        ## 输入实况real_seq和预报forecast_seq分别传入编码器
        # 实况encoder
        real_out = self.real_encoder(real_seq)
        
        # 预报encoder
        forecast_out = self.forecast_encoder(forecast_seq)
        
        B_r, C_r, L_r = real_out.shape
        B_f, C_f, L_f = forecast_out.shape
        
        # 交换维度
        real_out = real_out.permute(0, 2, 1)  # [batch, seq_len, channel]
        forecast_out = forecast_out.permute(0, 2, 1)
        
        # 统一通道维度
        real_out = self.blance_real(real_out)
        forecast_out = self.blance_forecast(forecast_out)
        
        # 位置编码
        real_out = self.position_encoding_real(real_out)
        forecast_out = self.position_encoding_forecast(forecast_out)
        
        # 使用BiLSTM处理实况数据
        real_lstm_out, _ = self.real_bilstm(real_out)
        real_lstm_out = self.real_dropout(real_lstm_out)
        real_lstm_out = self.real_layer_normal(real_out + real_lstm_out)  # 残差连接
        
        real_feed_out = self.real_ffn(real_lstm_out)
        real_lstm_out = self.real_feednorm(real_feed_out + real_lstm_out)  # 残差连接
        
        # 使用BiLSTM处理预报数据
        fcst_lstm_out, _ = self.fcst_bilstm(forecast_out)
        fcst_lstm_out = self.fcst_dropout(fcst_lstm_out)
        fcst_lstm_out = self.fcst_layer_norm(forecast_out + fcst_lstm_out)  # 残差连接
        
        # 交叉信息融合
        # 将实况和预报特征拼接在一起
        cross_input = torch.cat([fcst_lstm_out, real_lstm_out], dim=-1)
        
        # 交叉BiLSTM处理融合特征
        cross_lstm_out, _ = self.cross_bilstm(cross_input)
        cross_lstm_out = self.cross_dropout(cross_lstm_out)
        
        # 取交叉BiLSTM的输出，并进行残差连接和层归一化
        # 由于输入和输出维度不同，我们只使用BiLSTM的输出
        cross_lstm_out = self.cross_layer_norm(cross_lstm_out)
        
        # 使用前馈网络进一步处理
        cross_feed_out = self.cross_ffn(cross_lstm_out)
        cross_lstm_out = self.cross_feed_norm(cross_feed_out + cross_lstm_out)
        
        # 最终输出
        out = self.fc_out(cross_lstm_out)
        out = out.squeeze(dim=2)  # 压缩维度
        
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
            # 用 1x1 卷积进行通道数匹配
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
        
        # 初始的卷积层
        self.conv1 = nn.Conv1d(in_channel, 16, kernel_size=7, stride=1, padding=3)
        self.bn1 = nn.BatchNorm1d(16)
        self.relu = nn.ReLU(inplace=False)
        
        # ResNet 的残差块（可以叠加多个 BasicBlock）
        self.resblock1 = BasicBlock1D(16, out_channel)
        #self.resblock2 =BasicBlock1D(32,out_channel)

    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.resblock1(x)
        #x = self.resblock2(x)

        
        return x 