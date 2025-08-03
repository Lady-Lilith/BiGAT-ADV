import csv

import dgl
import dgl.function as fn
import dgl.nn.pytorch as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphConv


# from utils import to_etype_name

class TransformerRefinement(nn.Module):
    def __init__(self, d_model=256, nhead=4, num_layers=2):
        super(TransformerRefinement, self).__init__()
        self.embedding = nn.Linear(1, d_model)  # 变换到 d_model 维度
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True),
            num_layers=num_layers
        )
        self.fc = nn.Linear(d_model, 1)  # 变回 1 维

    def forward(self, x):
        x = self.embedding(x)  # (1300, 1) -> (1300, d_model)
        x = self.encoder(x)  # Transformer 编码
        x = self.fc(x)  # (1300, d_model) -> (1300, 1)
        return x


class HybridBidirectionalAttention(nn.Module):
    """双向交叉注意力 + 动态多头注意力"""

    def __init__(self, embed_dim, num_heads, dropout=0.1, res=True):
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.res = res

        # 交叉注意力
        self.cross_attn_ab2ag = nn.MultiheadAttention(embed_dim, num_heads, dropout)
        self.cross_attn_ag2ab = nn.MultiheadAttention(embed_dim, num_heads, dropout)

        # 动态多头注意力
        self.dynamic_attn_ab = DynamicMultiheadAttention(embed_dim, num_heads, dropout)
        self.dynamic_attn_ag = DynamicMultiheadAttention(embed_dim, num_heads, dropout)
        self.alpha_ab = nn.Parameter(torch.tensor(0.5))  # 初始化为 0.5
        self.alpha_ag = nn.Parameter(torch.tensor(0.5))  # 初始化为 0.5

        # 归一化
        self.norm_ab = nn.LayerNorm(embed_dim)
        self.norm_ag = nn.LayerNorm(embed_dim)

    def forward(self, antibody_embed, antigen_embed, antibody_mask=None, antigen_mask=None):
        """前向传播"""
        # 1️⃣ 交叉注意力
        attn_ab2ag, _ = self.cross_attn_ab2ag(antibody_embed, antigen_embed, antigen_embed,
                                              key_padding_mask=antigen_mask)
        attn_ag2ab, _ = self.cross_attn_ag2ab(antigen_embed, antibody_embed, antibody_embed,
                                              key_padding_mask=antibody_mask)

        # 2️⃣ 动态多头注意力
        dyn_attn_ab, _ = self.dynamic_attn_ab(antibody_embed, antibody_embed, antibody_embed,
                                              key_padding_mask=antibody_mask)
        dyn_attn_ag, _ = self.dynamic_attn_ag(antigen_embed, antigen_embed, antigen_embed,
                                              key_padding_mask=antigen_mask)

        alpha_ab_norm, alpha_ag_norm = torch.softmax(torch.stack([self.alpha_ab, self.alpha_ag]), dim=0)

        # 4️⃣ 动态融合注意力输出
        attn_output_ab = alpha_ab_norm.squeeze(0) * attn_ab2ag + alpha_ag_norm.squeeze(0) * dyn_attn_ab
        attn_output_ag = alpha_ab_norm.squeeze(0) * attn_ag2ab + alpha_ag_norm.squeeze(0) * dyn_attn_ag
        # 4️⃣ 残差连接 + 归一化
        if self.res:
            attn_output_ab = self.norm_ab(attn_output_ab + antibody_embed)
            attn_output_ag = self.norm_ag(attn_output_ag + antigen_embed)
        # if self.res:
        #     attn_output_ab = self.norm_ab(dyn_attn_ab + antibody_embed)
        #     attn_output_ag = self.norm_ag(dyn_attn_ag + antigen_embed)
        return attn_output_ab, attn_output_ag

class DynamicMultiheadAttention(nn.Module):
    """动态多头注意力"""

    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout)

        # 自适应权重 α (num_heads,)
        self.alpha = nn.Parameter(torch.ones(num_heads))

    def forward(self, query, key, value, key_padding_mask=None):
        attn_output, attn_weights = self.multihead_attn(query, key, value, key_padding_mask=key_padding_mask)

        # 归一化 α，使其 sum=1
        alpha_norm = F.softmax(self.alpha, dim=0).view(1, -1, 1, 1)  # (1, num_heads, 1, 1)

        # 让每个头的输出乘上 α 权重
        attn_output = attn_output * alpha_norm.sum(dim=1)  # (batch, seq_len, embed_dim)

        return attn_output, attn_weights


def read_csv(path):
    with open(path, 'r', newline='') as csv_file:
        reader = csv.reader(csv_file)
        cd_data = []
        cd_data += [[float(i) for i in row] for row in reader]
        return torch.Tensor(cd_data)
def to_etype_name(rating):
    return str(rating).replace(".", "_")
class HGATLinkConv(nn.Module):
    def __init__(
            self, args,in_feats,k_feats, out_feats,heads,d_k,weight=True,weight_k=True,relation_att=True, device=None, dropout_rate=0.0
        ,gate_net = False,  gate_hidden = 16,gate_scalar=False ): # 新增：门控网络隐藏层大小 ):
        super(HGATLinkConv, self).__init__()
        self.args = args
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._k_feats = k_feats
        self.device = device
        self.dropout = nn.Dropout(dropout_rate)
        self.n_heads = heads
        self.d_k = d_k
        self.dnum = args.disease_number
        if weight:
            self.weight = nn.Parameter(torch.Tensor(in_feats, out_feats))
        else:
            self.register_parameter("weight", None)
        if weight_k:
            self.weight_k = nn.Parameter(torch.Tensor(k_feats, out_feats))
        else:
            self.register_parameter("weight_k", None)
        self.linear_cir = nn.Linear(in_features=args.circRNA_number, out_features=args.out_channels)
        self.linear_dis = nn.Linear(in_features=args.disease_number, out_features=args.out_channels)
        self.reset_parameters()
        self.bicrossatt = bicrossatt(args, antibody_hidden_dim=args.out_channels, latent_dim=out_feats)
        self.cj_proj = nn.Linear(args.out_channels, args.msg)
        if gate_scalar:
            # 可学习标量 α，初始为 1.0
            self.alpha = nn.Parameter(torch.tensor(1.0))
        else:
            self.register_parameter('alpha', None)
        if gate_net:
            # 输入是单条边的预测分值 w_ij (shape [E,1])，输出一个 (0,1) 之间的门控系数
            layer0 = nn.Linear(1, gate_hidden, bias=True)
            nn.init.constant_(layer0.bias, 2.0)  # bias=2 -> sigmoid(2)≈0.88
            layer1 = nn.Linear(gate_hidden, 1, bias=True)
            self.gate_nn = nn.Sequential(
                layer0,
                nn.ReLU(),
                layer1,
                nn.Sigmoid()
            )
        else:
            self.gate_nn = None
    def reset_parameters(self):
        """Reinitialize parameters."""
        if self.weight is not None:
            nn.init.xavier_uniform_(self.weight)
            nn.init.xavier_uniform_(self.weight_k)

    def forward(self,graph, feat,weight=None,weight_k=None):
        with graph.local_scope():
            feat = feat if not isinstance(feat, tuple) else feat[0]
            # cj, ci = graph.srcdata["features"], graph.dstdata["features"]
            ci_path = self.args.dataset_path + '/c_c.csv'
            cj_path = self.args.dataset_path + '/d_d.csv'
            ci = read_csv(ci_path)
            cj = read_csv(cj_path)
            ci = self.linear_cir(ci)
            cj = self.linear_dis(cj)

            if self.device is not None:
                cj = cj
                ci = ci
            if weight is not None:
                if self.weight is not None:
                    raise dgl.DGLError(
                        "External weight is provided while at the same time the"
                        " module has defined its own weight parameter. Please"
                        " create the module with flag weight=False."
                    )
            else:
                weight = self.weight
                weight_k = self.weight_k
            if weight_k.shape[0] ==  self.dnum:
                ci, cj = cj, ci
            else:
                ci, cj = ci, cj
            res = feat
            if weight is not None:
                #1111111111111111111111111111111111111111
                feat = dot_or_identity(feat, weight_k, self.device)
                # weight_k = dot_or_identity(res, weight_k, self.device)
            attn, circ, dis = self.bicrossatt(ci, cj)
            cj = self.cj_proj(cj)
            ci = self.cj_proj(ci)
            #11111111111111111111111111111111111
            feat = F.relu(feat * self.dropout(ci))

            graph.srcdata["h"] = feat

            def message(edges):
                m = edges.src['h']  # [E, F]
                if 'weight' in edges.data:
                    w = edges.data['weight'].unsqueeze(1)  # [E,1]
                    # if self.gate_nn is not None:
                    if self.alpha is not None:
                        m = m * w * self.alpha
                        # 用小型 MLP 计算门控系数
                        g = self.gate_nn(w)  # [E,1] in (0,1)
                        m = m * g
                    else:
                        # 直接乘预测分值
                        m = m * w
                return {'m': m}

            # def message(edges):
            #     m = edges.src['h']  # [E, F]
            #     if 'weight' in edges.data:
            #         w = edges.data['weight'].unsqueeze(1)  # [E,1]
            #         m = m * w
            #     return {'m': m}

            graph.update_all(message, fn.max('m', 'h_new'))
            rst = graph.dstdata['h_new']
            # graph.update_all(
            #     fn.copy_u(u="h", out="m"), fn.max(msg="m", out="h")
            # )

            # rst = graph.dstdata["h"]
#111111111111111111111111111111111
            out = rst*dis
        return out
def dot_or_identity(A, B, device=None):
    # if A is None, treat as identity matrix
    if A is None:
        return B
    elif len(A.shape) == 1:
        if device is None:
            return B[A]
        else:
            return B[A].to(device)
    else:
        return A * B

class DGLLayer(nn.Module):
    def __init__(
            self,
            args,
            rating_vals,
            gene_in_units,
            cell_in_units,
            msg_units,
            out_units,
            dropout_rate=0.0,

            device=None,
    ):
        super(DGLLayer, self).__init__()
        self.args = args
        self.rating_vals = rating_vals

        self.ufc = nn.Linear(msg_units, out_units)
        self.ufc_adv = nn.Linear(msg_units*2, out_units)
        self.ifc = nn.Linear(msg_units, out_units)

        assert msg_units % len(rating_vals) == 0

        msg_units = msg_units // len(rating_vals)

        self.dropout = nn.Dropout(dropout_rate)
        self.W_r = nn.ParameterDict()
        # Steps to Increase Attention Mechanisms Dimensional Consistency on Targets
        self.W_k = nn.ParameterDict()
        subConv = {}
        heads, d_k = 4, 64
        self.etypes = [to_etype_name(r) for r in rating_vals] + ['pred_low']
        self.single_direction_types = ['pred_low']
        for rating in self.etypes:
            etype = to_etype_name(rating)  # 不覆盖 rating 原值
            rev_etype = "rev-" + etype
            # rev_rating = "rev-%s" % rating
            # Multiple Head Attention Mechanisms
            self.W_r = None
            self.W_k = None
            is_pred = (etype == 'pred_low')
            subConv[etype] = HGATLinkConv(
                args,
                gene_in_units,
                cell_in_units,
                msg_units,
                heads,
                d_k,
                weight=True,
                weight_k=True,
                device=device,
                dropout_rate=dropout_rate,
                gate_net=is_pred,
                gate_hidden=16,
                gate_scalar=is_pred
            )
            if etype not in self.single_direction_types:
                # rev_rating = "rev-" + rating
                subConv[rev_etype] = HGATLinkConv(
                    args,
                    cell_in_units,
                    gene_in_units,
                    msg_units,
                    heads,
                    d_k,
                    weight=True,
                    weight_k=True,
                    device=device,
                    dropout_rate=dropout_rate,
                    gate_net=False,
                    gate_hidden=16,
                    gate_scalar=False
                )

        self.conv = dglnn.HeteroGraphConv(subConv, aggregate='stack')
        self.agg_act = nn.ReLU()
        self.out_act = lambda x: x
        self.device = device
        self.reset_parameters()

    def partial_to(self, device):

        assert device == self.device
        if device is not None:
            self.ufc.cuda(device)
            if self.share_gene_item_param is False:
                self.ifc.cuda(device)
            self.dropout.cuda(device)

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, args,graph, ufeat=None, ifeat=None):
        in_feats = {"rna": ufeat, "disease": ifeat}

        mod_args = {}
        for i, rating in enumerate(self.rating_vals):
            rating = to_etype_name(rating)
            rev_rating = "rev-%s" % rating
            mod_args[rating] = (
                self.W_r[rating] if self.W_r is not None else None,
            )
            mod_args[rev_rating] = (
                self.W_r[rev_rating] if self.W_r is not None else None,
            )

        # 1071*15*256  704*15*256
        out_feats = self.conv(graph, in_feats, mod_args=mod_args)
        # 获取特定类型的边

        ufeat = out_feats.get("rna")
        ifeat = out_feats.get("disease")

        # 如果 "rna" 特征为空，则使用原始输入特征
        if ufeat is None or ufeat.shape[0] == 0:
            ufeat = self.rna_emb.weight  # (num_rna, gene_in_units)
        if ifeat is None:
            ifeat = self.dis_emb.weight  # (num_disease, cell_in_units)


        '''ufeat = out_feats["rna"]
        ifeat = out_feats["disease"]'''

        ufeat = ufeat.view(ufeat.shape[0], -1)

        ifeat = ifeat.view(ifeat.shape[0], -1)

        # fc and non-linear
        ufeat = self.agg_act(ufeat)
        ifeat = self.agg_act(ifeat)
        ufeat = self.dropout(ufeat)
        ifeat = self.dropout(ifeat)
        if ufeat.shape[-1] == 512:
            ufeat = self.ufc(ufeat)
        else:
            ufeat = self.ufc_adv(ufeat)
        ifeat = self.ifc(ifeat)
        ufeat = self.out_act(ufeat)
        ifeat = self.out_act(ifeat)

        return ufeat.mm(ifeat.t()),ufeat, ifeat

class Decoder(nn.Module):
    def __init__(self,args, dropout_rate=0.0):
        super(Decoder, self).__init__()
        self.args = args
        self.dropout = nn.Dropout(dropout_rate)
        # self.cross = bicrossatt(args,antibody_hidden_dim=256, latent_dim = 128, res = False)
    def forward(self, graph, ufeat, ifeat):
        with graph.local_scope():
            ufeat, ifeat = self.dropout(ufeat), self.dropout(ifeat)
            graph.nodes["disease"].data["h"] = ifeat
            graph.nodes["rna"].data["h"] = ufeat
            graph.apply_edges(fn.u_dot_v("h", "h", "sr"))
            # graph.edata['sr'] = edge_features
            # x = self.gnn(graph, graph.edata['sr'])
            # x = self.transformer(graph.edata['sr'])
            return graph.edata['sr'], graph.nodes["rna"].data["h"], graph.nodes["disease"].data["h"]

class bicrossatt(nn.Module):
    def __init__(self, args,antibody_hidden_dim=64, latent_dim = 128, ) -> None:
        super().__init__()
        self.args = args
        # self.bidirectional_crossatt = BidirectionalCrossAttention(embed_dim=antibody_hidden_dim, num_heads=1,res=res)
        self.bidirectional_crossatt = HybridBidirectionalAttention(embed_dim=antibody_hidden_dim, num_heads=1)
        self.LayerNorm = nn.LayerNorm(antibody_hidden_dim, eps=1e-12)
        self.linear = nn.Sequential(nn.Linear(antibody_hidden_dim,antibody_hidden_dim),nn.ReLU(inplace=True))
        self.change_dim = nn.Sequential(nn.Linear(antibody_hidden_dim,latent_dim),nn.ReLU(inplace=True))
        self.alpha = nn.Parameter(torch.tensor([1.0]))

    def forward(self, x_cir,x_dis):

        antibody_seq_stru,antigen_seq_stru = self.bidirectional_crossatt(x_cir,x_dis)
        antibody_seq_stru = self.change_dim(self.linear(self.LayerNorm(antibody_seq_stru)))
        antigen_seq_stru = self.change_dim(self.linear(self.LayerNorm(antigen_seq_stru)))
        antibody_seq_stru = antibody_seq_stru.squeeze(0)
        antigen_seq_stru = antigen_seq_stru.squeeze(0)
        concatenated_tensor = torch.matmul(antibody_seq_stru, (self.alpha*antigen_seq_stru).T)
        return torch.sigmoid(concatenated_tensor),antibody_seq_stru,antigen_seq_stru

class pool(nn.Module):
    def __init__(self, latent_dim) -> None:
        super().__init__()
        self.linear1 = nn.Sequential(nn.Linear( latent_dim, latent_dim * latent_dim), nn.ReLU(inplace=True))
        self.linear2 = nn.Sequential(nn.Linear( latent_dim, latent_dim * latent_dim), nn.ReLU(inplace=True))

    def forward(self, input, latent_dim, is_antibody=True):
        batch_size = input.size(0)
        input = input.view(batch_size, -1)
        if is_antibody:
            output = self.linear1(input)
        else:
            output = self.linear2(input)
        output_tensor = output.view(batch_size, latent_dim, latent_dim)
        return output_tensor


class BidirectionalCrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1, res=False):
        super().__init__()
        # Define multi-head attention Layers for both directions
        self.antibody_to_antigen_attention = nn.MultiheadAttention(256, num_heads, dropout, bias=True,
                                                                   add_bias_kv=False)
        self.antigen_to_antibody_attention = nn.MultiheadAttention(256, num_heads, dropout, bias=True,
                                                                   add_bias_kv=False)

        # Optional: You can add layer normalization if needed
        self.antibody_norm = nn.LayerNorm(embed_dim)
        self.antigen_norm = nn.LayerNorm(embed_dim)
        self.res = res
    def forward(self, antibody_embed, antigen_embed, antibody_mask=None, antigen_mask=None):
        # Antibody to Antigen Attention
        antibody_as_query = antibody_embed
        antigen_as_kv = antigen_embed
        attn_output_antibody, attn_weights_antibody = self.antibody_to_antigen_attention(
            query=antibody_as_query,
            key=antigen_as_kv,
            value=antigen_as_kv,
            key_padding_mask=antigen_mask
        )
        # Antigen to Antibody Attention
        antigen_as_query = antigen_embed
        antibody_as_kv = antibody_embed
        attn_output_antigen, attn_weights_antigen = self.antigen_to_antibody_attention(
            query=antigen_as_query,
            key=antibody_as_kv,
            value=antibody_as_kv,
            key_padding_mask=antibody_mask
        )
        # residual connection(optional)
        if self.res == True:
            # Residual connection for antibody to antigen attention
            attn_output_antibody = attn_output_antibody + antibody_as_query
            attn_output_antibody = self.antibody_norm(attn_output_antibody)  # Optional normalization
            # Residual connection for antigen to antibody attention
            attn_output_antigen = attn_output_antigen + antigen_as_query
            attn_output_antigen = self.antigen_norm(attn_output_antigen)  # Optional normalization
        return attn_output_antibody, attn_output_antigen



class LinearNet(nn.Module):
    def __init__(self, args,emb_dim=256, num_heads=4, num_layers=2, dropout_rate=0.5):
        super(LinearNet, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=emb_dim, nhead=num_heads, batch_first=True),
            num_layers=num_layers
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=emb_dim, nhead=num_heads, dim_feedforward=emb_dim, dropout=dropout_rate),
            num_layers=1
        )
        self.flatten = nn.Flatten()
        # self.linear256 = nn.Linear(512, 256)
        self.batchnorm256 = nn.BatchNorm1d(emb_dim)
        self.linear2 = nn.Linear(emb_dim, 1)

    def forward(self, x):
        x = self.dropout(x)
        out = self.encoder(x)
        out = self.decoder(x, out)
        out = self.flatten(out)
        out = self.batchnorm256(out)
        return F.gelu(out)
class DGLLayerHomogeneous(nn.Module):
    def __init__(
        self,
        args,
        gene_in_units,
        cell_in_units,
        msg_units,   # 同构卷积输出的维度（平铺所有注意力头后的维度）
        out_units,
        dropout_rate=0.0,
        device=None,
    ):
        super(DGLLayerHomogeneous, self).__init__()
        self.args = args
        self.device = device

        # 将 RNA 和 disease 映射到同一维度（common_dim），便于拼接
        # 这里 common_dim 可以设置为 gene_in_units 或 cell_in_units 的最大值，或者自行指定
        self.common_dim = max(gene_in_units, cell_in_units)
        self.rna_proj = nn.Linear(585, max(gene_in_units, cell_in_units))
        self.dis_proj = nn.Linear(88, max(gene_in_units, cell_in_units))

        # 使用同构图卷积，这里以 GATConv 为例
        heads = 4
        # GATConv 的输出维度为 out_dim，每个头独立，拼接后总维度为 heads*out_dim
        # 为使最终维度等于 msg_units，我们令 out_dim = msg_units // heads
        self.homo_conv = dglnn.GATConv(
            in_feats=self.common_dim,
            out_feats=msg_units // heads,
            num_heads=heads,
            feat_drop=dropout_rate,
            attn_drop=dropout_rate,
            activation=F.elu
        )

        # 后续全连接层，分别对 RNA 和 disease 部分进行变换
        self.ufc = nn.Linear(msg_units, out_units)
        self.ifc = nn.Linear(msg_units, out_units)
        self.rna_emb = nn.Embedding(585, gene_in_units)
        self.dis_emb = nn.Embedding(88, cell_in_units)
        self.agg_act = nn.ReLU()
        self.out_act = lambda x: x
        self.dropout = nn.Dropout(dropout_rate)

        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, graph, ufeat, ifeat):
        """
        参数说明：
          graph: 同构图（dgl.DGLGraph），节点顺序要求与 ufeat, ifeat 拼接顺序一致。
          ufeat: RNA 节点特征，形状 (num_rna, gene_in_units)
          ifeat: disease 节点特征，形状 (num_disease, cell_in_units)
        返回：
          score矩阵、RNA 变换后的特征、disease 变换后的特征
        """
        # 分别投影到 common_dim
        if ufeat is None:
            ufeat = self.rna_emb.weight
            rna_feat = self.rna_proj(ufeat) # (num_rna, gene_in_units)
        else:
            rna_feat = ufeat
        if ifeat is None:
            ifeat = self.dis_emb.weight
            dis_feat = self.dis_proj(ifeat)
        else:# (num_disease, cell_in_units)    # (num_disease, common_dim)
            dis_feat = ifeat
        # 拼接成一个同构图的节点特征
        combined_feat = torch.cat([rna_feat, dis_feat], dim=0)  # (num_total, common_dim)

        # 同构图卷积：输出形状 (num_total, num_heads, out_dim)
        conv_out = self.homo_conv(graph, combined_feat)
        # 将多头拼接成一个向量 (num_total, msg_units)
        conv_out = conv_out.flatten(1)

        # 根据 RNA 节点数量拆分结果
        num_rna = ufeat.shape[0]
        ufeat_out = conv_out[:num_rna]      # RNA 特征
        ifeat_out = conv_out[num_rna:]      # disease 特征

        # 后续处理：激活、dropout、全连接
        ufeat_out = self.agg_act(ufeat_out)
        ifeat_out = self.agg_act(ifeat_out)

        ufeat_out = self.dropout(ufeat_out)
        ifeat_out = self.dropout(ifeat_out)

        ufeat_out = self.ufc(ufeat_out)
        ifeat_out = self.ifc(ifeat_out)

        ufeat_out = self.out_act(ufeat_out)
        ifeat_out = self.out_act(ifeat_out)

        # 返回打分矩阵（例如点乘得分）以及各自的特征
        score = ufeat_out.mm(ifeat_out.t())
        return score, ufeat_out, ifeat_out