import torch.nn as nn
from models import iTransformer_block
from models.TCN import TemporalConvNet
import torch
from fightingcv_attention.attention.ExternalAttention import ExternalAttention


class TCN_iTransformer(nn.Module):
    def __init__(self, input_size=5, num_channels=[64] * 3, kernel_size=3, num_classes=10,
                 length_MC=10, dim_embed=512, depth=6, heads=8, dim_mlp=20, dim_head=128):
        super(TCN_iTransformer, self).__init__()
        self.model1 = iTransformer_block(num_variates=input_size, lookback_len=length_MC,
                                         heads=8, dim_head=64, pred_length=(10), num_class=num_classes, dim=dim_embed, depth=depth,
                                         num_tokens_per_variate=1, use_reversible_instance_norm=True)
        self.model2 = TemporalConvNet(num_inputs=input_size, num_channels=num_channels,
                                      kernel_size=kernel_size, dropout=0.1)
        self.fc1 = nn.Linear(input_size, dim_mlp)
        self.fc2 = nn.Linear(num_channels[-1], dim_mlp)
        self.rl = nn.ReLU()
        # self.att = ExternalAttention(d_model=num_channels[-1], S=16)  # BxLxC-->BxLxC
        self.fc = nn.Linear(num_channels[-1] * length_MC, num_classes)
        self.mlp = nn.Sequential(
            nn.Linear(length_MC, dim_mlp),
            nn.ReLU(),
            nn.Linear(dim_mlp, num_classes)
        )
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=input_size,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=False)

    def forward(self, x):
        x, _ = self.lstm(x)
        x1 = self.model1(x)  # BxLxC
        x1 = self.model2(x1.transpose(2, 1))  # BxCxL
        # x1=self.rl(self.fc1(x1))
        # x2=self.rl(self.fc2(x2.transpose(2,1))) # BxLxC
        # combined_features = torch.cat((x1, x2), dim=-1)
        # f_att=self.att(combined_features)
        # output = self.fc(combined_features)
        # x1=self.att(x1.transpose(2,1))
        output = self.mlp(x1[:, -1, :])
        # output = self.fc(x1.flatten(1))
        return output


class iTransformer_TCN(nn.Module):
    def __init__(self, input_size=5, num_channels=[32] * 1, kernel_size=3, length_pre=1,
                 length_input=48, dim_embed=128, depth=4, heads=6, dim_mlp=10, dim_head=64):
        super(iTransformer_TCN, self).__init__()
        self.model1 = iTransformer_block(num_variates=5, lookback_len=48,
                                         pred_length=num_channels[0], dim=dim_embed, depth=depth,
                                         num_tokens_per_variate=1, use_reversible_instance_norm=True)
        self.model2 = TemporalConvNet(num_inputs=1, num_channels=num_channels,
                                      kernel_size=kernel_size, dropout=0.2)
        self.fc1 = nn.Linear(input_size, dim_mlp)
        self.fc2 = nn.Linear(num_channels[-1] * num_channels[0], dim_mlp)
        self.rl = nn.ReLU()
        # self.att = ExternalAttention(d_model=dim_mlp * 2, S=64)  # BxLxC-->BxLxC
        self.fc = nn.Linear(dim_embed, length_pre)
        self.mlp = nn.Sequential(
            nn.Linear(num_channels[-1] * dim_embed, dim_mlp),
            nn.ReLU(),
            # Densegauss(dim_mlp, length_pre),
            nn.Linear(dim_mlp, length_pre),
        )

    def forward(self, x):
        # x[:,:,0]=self.inv1(x[:,:,0], 'norm')
        # x[:, :, 1:] = self.inv2(x[:, :, 1:], 'norm')
        x = self.model1(x)
        x = x[:, 0, None, :]
        x = self.model2(x)  # BxLxC
        # x=self.fc(x)
        # BxCxL
        # x1=self.rl(self.fc1(x1))
        # x2=self.rl(self.fc2(x2.transpose(2,1))) # BxLxC
        # combined_features = torch.cat((x1, x2), dim=-1)
        # f_att=self.att(combined_features)
        # output = self.fc(combined_features)
        output = self.mlp(x.flatten(1))
        # output = self.inv1(output.unsqueeze(1), 'denorm')
        return output.flatten(1)


class parallel_iTransformer_TCN(nn.Module):
    def __init__(self, input_size=57, num_channels=[64] * 4, kernel_size=3, num_classes=10,
                 length_MC=10, dim_embed=512, depth=6, heads=8, dim_mlp=20, dim_head=128):
        super(parallel_iTransformer_TCN, self).__init__()
        self.model1 = iTransformer_block(num_variates=input_size, lookback_len=length_MC,
                                         heads=8, dim_head=64, pred_length=(10), num_class=num_classes, dim=dim_embed, depth=depth,
                                         num_tokens_per_variate=1, use_reversible_instance_norm=True)
        self.model2 = TemporalConvNet(num_inputs=input_size, num_channels=num_channels,
                                      kernel_size=kernel_size, dropout=0.2)
        self.fc1 = nn.Linear(input_size, dim_mlp)
        self.fc2 = nn.Linear(num_channels[-1], dim_mlp)
        self.rl = nn.ReLU()
        self.att = ExternalAttention(d_model=dim_mlp * 2, S=64)  # BxLxC-->BxLxC
        self.fc = nn.Linear(dim_mlp * length_MC, num_classes)
        self.mlp = nn.Sequential(
            nn.Linear(2 * dim_mlp * length_MC, dim_mlp),
            nn.ReLU(),
            nn.Linear(dim_mlp, num_classes)
        )

    def forward(self, x):
        x1 = self.model1(x)  # BxLxC
        x2 = self.model2(x.transpose(2, 1))  # BxCxL
        x1 = self.rl(self.fc1(x1))
        x2 = self.rl(self.fc2(x2.transpose(2, 1)))  # BxLxC
        x_f = torch.cat((x1, x2), dim=-1)
        # f_att=self.att(combined_features)
        # output = self.fc(combined_features)
        output = self.mlp(x_f.flatten(1))
        return output
