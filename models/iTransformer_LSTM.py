from models import iTransformer_block,CrossAttention
import torch.nn as nn
from models.KAN import KAN
class iTransformer_LSTM(nn.Module):
    def __init__(self, input_size=5,  length_pre=1,dim_lstm=128,depth_lstm=3,
                 length_input=48, dim_embed=128, depth=4, heads=6):
        super(iTransformer_LSTM, self).__init__()
        self.model1 = iTransformer_block(num_variates=1, lookback_len=length_input,
                                         pred_length=length_pre, dim=dim_embed, depth=depth, heads=heads,
                                         num_tokens_per_variate=1, use_reversible_instance_norm=True)
        # self.model2 = TemporalConvNet(num_inputs=input_size, num_channels=num_channels,
        #                               kernel_size=kernel_size, dropout=0.2)
        self.lstm = nn.LSTM(input_size=input_size-1,
                            hidden_size=dim_lstm,
                            num_layers=depth_lstm,
                            batch_first=True,
                            bidirectional=False)
        # self.fc1 = nn.Linear(input_size, dim_mlp)
        # self.fc2 = nn.Linear(num_channels[-1] * num_channels[0], dim_mlp)
        # self.rl = nn.ReLU()
        # # self.att = ExternalAttention(d_model=dim_mlp * 2, S=64)  # BxLxC-->BxLxC
        # self.fc = nn.Linear(dim_embed, dim_embed)
        # ratio_mlp = 2
        # self.mlp_t = nn.Sequential(
        #     nn.Linear(length_input * num_channels[-1], dim_mlp * ratio_mlp),
        #     nn.ReLU(),
        #     nn.Linear(dim_mlp * ratio_mlp, length_pre),
        #     nn.ReLU(),
        # )
        self.mlp_i = nn.Sequential(
            nn.Linear(dim_embed, length_pre),
            # nn.ReLU(),
            # nn.Linear(dim_mlp * ratio_mlp, length_pre),
            # nn.ReLU(),
        )
        self.k_mpl=KAN([dim_embed, length_pre])
        # self.fc1 = nn.Sequential(nn.Linear(input_size, 1),
        #                          nn.ReLU(),
        #                          )
        # self.mlp = nn.Sequential(
        #     nn.Linear(input_size * dim_embed, dim_mlp * ratio_mlp),
        #     nn.ReLU(),
        #     nn.Linear(dim_mlp * ratio_mlp, dim_mlp),
        #     nn.ReLU(),
        #     nn.Linear(dim_mlp, length_pre),
        # )
        self.cross = CrossAttention(dim=dim_embed, lenth=dim_lstm)
        # self.inv1 = RevIN(num_features=1)
        # self.inv2 = RevIN(num_features=input_size - 1)

    def forward(self, x):
        # x[:,:,0]=self.inv1(x[:,:,0],'norm')
        # x[:, :, 1:] = self.inv2(x[:, :, 1:],'norm')

        x2,_ = self.lstm(x[:,:,1:])
        x1 = self.model1(x[:,:,0,None])
        x1=self.cross(x1,x2)
        # output = self.fc1(x1.transpose(2, 1)).flatten(1)
        # output = self.mlp_i(x1)
        output=self.k_mpl(x1)

        # output = output.flatten(1)
        # output=self.mlp(out.flatten(1))
        # x1 = self.mlp_i(x1.flatten(1))
        # x2 = self.mlp_t(x2.flatten(1))
        # output = self.mlp(torch.concatenate([x1, x2], dim=1))
        # output=self.inv1(output.unsqueeze(1),'denorm')
        # BxLxC
        # x=self.fc(x)
        # BxCxL
        # x1=self.rl(self.fc1(x1))
        # x2=self.rl(self.fc2(x2.transpose(2,1))) # BxLxC
        # combined_features = torch.cat((x1, x2), dim=-1)
        # f_att=self.att(combined_features)
        # output = self.fc(combined_features)
        # output = self.mlp(x.flatten(1))
        return output[:,0,:]


