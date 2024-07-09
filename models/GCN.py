#-------------------------------------------------------------------------------
# Name:        GCN.py
# Purpose:     definition of joint prediction module.
# RigNet Copyright 2020 University of Massachusetts
# RigNet is made available under General Public License Version 3 (GPLv3), or under a Commercial License.
# Please see the LICENSE README.txt file in the main directory for more information and instruction on using and licensing RigNet.
#-------------------------------------------------------------------------------
import torch
from models.gcn_basic_modules import MLP, GCU
from torch_scatter import scatter_max
from torch.nn import Linear

class TARigPrimaryJointPredNet(torch.nn.Module):
    def __init__(self, input_normal, num_joints, attention_weights_activation, aggr='max'):
        '''
        Parameters:
            input_normal: whether to also use vertex normal vectors as input
            num_joints: number of joints (output channels)
            attention_weights_activation: activation function for attention weights ("sigmoid" or "softmax")
            aggr: aggregation method for GCU ("max" or "mean")
        '''
        super(TARigPrimaryJointPredNet, self).__init__()
        self.input_normal = input_normal
        self.num_joints = num_joints
        self.input_channels = 6 if input_normal else 3
        self.attention_weights_activation = attention_weights_activation

        self.gcu_1 = GCU(in_channels=self.input_channels, out_channels=64, aggr=aggr)
        self.gcu_2 = GCU(in_channels=64, out_channels=256, aggr=aggr)
        self.gcu_3 = GCU(in_channels=256, out_channels=512, aggr=aggr)
        
        # feature compression
        self.mlp_glb = MLP([(64 + 256 + 512), 1024])

        # (vertex, joint) influence prediction
        self.mlp_transform = Linear(1024 + self.input_channels + 64 + 256 + 512, num_joints)


    def forward(self, data):
        x = torch.cat([data.pos, data.x], dim=1) if self.input_normal else data.pos
        geo_edge_index, tpl_edge_index, batch = data.geo_edge_index, data.tpl_edge_index, data.batch

        x_1 = self.gcu_1(x, tpl_edge_index, geo_edge_index)
        x_2 = self.gcu_2(x_1, tpl_edge_index, geo_edge_index)
        x_3 = self.gcu_3(x_2, tpl_edge_index, geo_edge_index)
        x_4 = self.mlp_glb(torch.cat([x_1, x_2, x_3], dim=1))

        x_global, _ = scatter_max(x_4, data.batch, dim=0)
        x_global = torch.repeat_interleave(x_global, torch.bincount(data.batch), dim=0)

        x_5 = torch.cat([x_global, x, x_1, x_2, x_3], dim=1)
        out = self.mlp_transform(x_5)

        assert out.shape == (data.pos.shape[0], self.num_joints)
        batch_size = len(torch.unique(data.batch))

        attention_weights_list = []

        if self.attention_weights_activation == 'softmax':
            outs = []
            for i in range(batch_size):
                # We need to apply a softmax to each column to get per-vertex weights for each joint, such that they sum to 1.
                attention_weights = torch.softmax(out[data.batch == i], dim=0)
                attention_weights_list.append(attention_weights)

                # Derive keypoint locations as weighted sums of the points
                out_i = attention_weights.transpose(-1, -2) @ data.pos[data.batch == i]
                
                outs.append(out_i)
            out = outs
        elif self.attention_weights_activation == 'sigmoid':
            out = torch.sigmoid(out)
            outs = torch.empty((batch_size, self.num_joints, 3)).to(out.device)
            for i in range(batch_size):
                out_i = out[data.batch == i]
                out_i = out_i / (torch.sum(out_i, dim=0) + 1e-9)
                attention_weights_list.append(out_i)
                outs[i] = out_i.transpose(-1, -2) @ data.pos[data.batch == i]
        
        out = outs
        return out, attention_weights_list