"""MotionAGFormer model scripts are copy of repository originally released under the Apache-2.0 License.
Details of the original repository is as follows:
- Original repository: https://github.com/TaatiTeam/MotionAGFormer
"""

import math
from collections import OrderedDict

import torch
import torch.nn.functional as F
from timm.layers import DropPath
from torch import nn

from bidipose.models.base import BaseModel

from .modules.attention import Attention
from .modules.graph import GCN
from .modules.mlp import MLP
from .modules.tcn import MultiScaleTCN


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal position embeddings for timestep encoding."""

    def __init__(self, dim: int):
        """Initialize sinusoidal position embeddings.

        Args:
            dim (int): Embedding dimension.

        """
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """Forward pass for sinusoidal position embeddings.

        Args:
            time (torch.Tensor): Timestep tensor with shape (B,).

        Returns:
            torch.Tensor: Sinusoidal embeddings with shape (B, dim).

        """
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class AGFormerBlock(nn.Module):
    """Implementation of AGFormer block."""

    def __init__(
        self,
        dim,
        mlp_ratio=4.0,
        act_layer=nn.GELU,
        attn_drop=0.0,
        drop=0.0,
        drop_path=0.0,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        use_layer_scale=True,
        layer_scale_init_value=1e-5,
        mode="spatial",
        mixer_type="attention",
        use_temporal_similarity=True,
        temporal_connection_len=1,
        neighbour_num=4,
        n_frames=243,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        if mixer_type == "attention":
            self.mixer = Attention(
                dim,
                dim,
                num_heads,
                qkv_bias,
                qk_scale,
                attn_drop,
                proj_drop=drop,
                mode=mode,
            )
        elif mixer_type == "graph":
            self.mixer = GCN(
                dim,
                dim,
                num_nodes=17 if mode == "spatial" else n_frames,
                neighbour_num=neighbour_num,
                mode=mode,
                use_temporal_similarity=use_temporal_similarity,
                temporal_connection_len=temporal_connection_len,
            )
        elif mixer_type == "ms-tcn":
            self.mixer = MultiScaleTCN(in_channels=dim, out_channels=dim)
        else:
            raise NotImplementedError("AGFormer mixer_type is either attention or graph")
        self.norm2 = nn.LayerNorm(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        # The following two techniques are useful to train deep GraphFormers.
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)

    def forward(self, x):
        """x: tensor with shape [B, T, J, C]"""
        if self.use_layer_scale:
            x = x + self.drop_path(self.layer_scale_1.unsqueeze(0).unsqueeze(0) * self.mixer(self.norm1(x)))
            x = x + self.drop_path(self.layer_scale_2.unsqueeze(0).unsqueeze(0) * self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.mixer(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class MotionAGFormerBlock(nn.Module):
    """Implementation of MotionAGFormer block. It has two ST and TS branches followed by adaptive fusion."""

    def __init__(
        self,
        dim,
        mlp_ratio=4.0,
        act_layer=nn.GELU,
        attn_drop=0.0,
        drop=0.0,
        drop_path=0.0,
        num_heads=8,
        use_layer_scale=True,
        qkv_bias=False,
        qk_scale=None,
        layer_scale_init_value=1e-5,
        use_adaptive_fusion=True,
        hierarchical=False,
        use_temporal_similarity=True,
        temporal_connection_len=1,
        use_tcn=False,
        graph_only=False,
        neighbour_num=4,
        n_frames=243,
    ):
        super().__init__()
        self.hierarchical = hierarchical
        dim = dim // 2 if hierarchical else dim

        # ST Attention branch
        self.att_spatial = AGFormerBlock(
            dim,
            mlp_ratio,
            act_layer,
            attn_drop,
            drop,
            drop_path,
            num_heads,
            qkv_bias,
            qk_scale,
            use_layer_scale,
            layer_scale_init_value,
            mode="spatial",
            mixer_type="attention",
            use_temporal_similarity=use_temporal_similarity,
            neighbour_num=neighbour_num,
            n_frames=n_frames,
        )
        self.att_temporal = AGFormerBlock(
            dim,
            mlp_ratio,
            act_layer,
            attn_drop,
            drop,
            drop_path,
            num_heads,
            qkv_bias,
            qk_scale,
            use_layer_scale,
            layer_scale_init_value,
            mode="temporal",
            mixer_type="attention",
            use_temporal_similarity=use_temporal_similarity,
            neighbour_num=neighbour_num,
            n_frames=n_frames,
        )

        # ST Graph branch
        if graph_only:
            self.graph_spatial = GCN(dim, dim, num_nodes=17, mode="spatial")
            if use_tcn:
                self.graph_temporal = MultiScaleTCN(in_channels=dim, out_channels=dim)
            else:
                self.graph_temporal = GCN(
                    dim,
                    dim,
                    num_nodes=n_frames,
                    neighbour_num=neighbour_num,
                    mode="temporal",
                    use_temporal_similarity=use_temporal_similarity,
                    temporal_connection_len=temporal_connection_len,
                )
        else:
            self.graph_spatial = AGFormerBlock(
                dim,
                mlp_ratio,
                act_layer,
                attn_drop,
                drop,
                drop_path,
                num_heads,
                qkv_bias,
                qk_scale,
                use_layer_scale,
                layer_scale_init_value,
                mode="spatial",
                mixer_type="graph",
                use_temporal_similarity=use_temporal_similarity,
                temporal_connection_len=temporal_connection_len,
                neighbour_num=neighbour_num,
                n_frames=n_frames,
            )
            self.graph_temporal = AGFormerBlock(
                dim,
                mlp_ratio,
                act_layer,
                attn_drop,
                drop,
                drop_path,
                num_heads,
                qkv_bias,
                qk_scale,
                use_layer_scale,
                layer_scale_init_value,
                mode="temporal",
                mixer_type="ms-tcn" if use_tcn else "graph",
                use_temporal_similarity=use_temporal_similarity,
                temporal_connection_len=temporal_connection_len,
                neighbour_num=neighbour_num,
                n_frames=n_frames,
            )

        self.use_adaptive_fusion = use_adaptive_fusion
        if self.use_adaptive_fusion:
            self.fusion = nn.Linear(dim * 2, 2)
            self._init_fusion()

    def _init_fusion(self):
        self.fusion.weight.data.fill_(0)
        self.fusion.bias.data.fill_(0.5)

    def forward(self, x):
        """x: tensor with shape [B, T, J, C]"""
        if self.hierarchical:
            B, T, J, C = x.shape
            x_attn, x_graph = x[..., : C // 2], x[..., C // 2 :]

            x_attn = self.att_temporal(self.att_spatial(x_attn))
            x_graph = self.graph_temporal(self.graph_spatial(x_graph + x_attn))
        else:
            x_attn = self.att_temporal(self.att_spatial(x))
            x_graph = self.graph_temporal(self.graph_spatial(x))

        if self.hierarchical:
            x = torch.cat((x_attn, x_graph), dim=-1)
        elif self.use_adaptive_fusion:
            alpha = torch.cat((x_attn, x_graph), dim=-1)
            alpha = self.fusion(alpha)
            alpha = alpha.softmax(dim=-1)
            x = x_attn * alpha[..., 0:1] + x_graph * alpha[..., 1:2]
        else:
            x = (x_attn + x_graph) * 0.5

        return x


def create_layers(
    dim,
    n_layers,
    mlp_ratio=4.0,
    act_layer=nn.GELU,
    attn_drop=0.0,
    drop_rate=0.0,
    drop_path_rate=0.0,
    num_heads=8,
    use_layer_scale=True,
    qkv_bias=False,
    qkv_scale=None,
    layer_scale_init_value=1e-5,
    use_adaptive_fusion=True,
    hierarchical=False,
    use_temporal_similarity=True,
    temporal_connection_len=1,
    use_tcn=False,
    graph_only=False,
    neighbour_num=4,
    n_frames=243,
):
    """Generates MotionAGFormer layers"""
    layers = []
    for _ in range(n_layers):
        layers.append(
            MotionAGFormerBlock(
                dim=dim,
                mlp_ratio=mlp_ratio,
                act_layer=act_layer,
                attn_drop=attn_drop,
                drop=drop_rate,
                drop_path=drop_path_rate,
                num_heads=num_heads,
                use_layer_scale=use_layer_scale,
                layer_scale_init_value=layer_scale_init_value,
                qkv_bias=qkv_bias,
                qk_scale=qkv_scale,
                use_adaptive_fusion=use_adaptive_fusion,
                hierarchical=hierarchical,
                use_temporal_similarity=use_temporal_similarity,
                temporal_connection_len=temporal_connection_len,
                use_tcn=use_tcn,
                graph_only=graph_only,
                neighbour_num=neighbour_num,
                n_frames=n_frames,
            )
        )
    layers = nn.Sequential(*layers)

    return layers


class MotionAGFormer(BaseModel):
    """MotionAGFormer, the main class of our model."""

    def __init__(
        self,
        n_layers,
        dim_feat,
        dim_in=6,
        dim_rep=512,
        dim_out=6,
        mlp_ratio=4,
        act_layer=nn.GELU,
        attn_drop=0.0,
        drop=0.0,
        drop_path=0.0,
        use_layer_scale=True,
        layer_scale_init_value=1e-5,
        use_adaptive_fusion=True,
        num_heads=4,
        qkv_bias=False,
        qkv_scale=None,
        hierarchical=False,
        num_joints=17,
        use_temporal_similarity=True,
        temporal_connection_len=1,
        use_tcn=False,
        graph_only=False,
        neighbour_num=4,
        n_frames=88,
        timestep_embed_dim: int = 64,
    ):
        """:param n_layers: Number of layers.
        :param dim_in: Input dimension.
        :param dim_feat: Feature dimension.
        :param dim_rep: Motion representation dimension
        :param dim_out: output dimension. For 3D pose lifting it is set to 3
        :param mlp_ratio: MLP ratio.
        :param act_layer: Activation layer.
        :param drop: Dropout rate.
        :param drop_path: Stochastic drop probability.
        :param use_layer_scale: Whether to use layer scaling or not.
        :param layer_scale_init_value: Layer scale init value in case of using layer scaling.
        :param use_adaptive_fusion: Whether to use adaptive fusion or not.
        :param num_heads: Number of attention heads in attention branch
        :param qkv_bias: Whether to include bias in the linear layers that create query, key, and value or not.
        :param qkv_scale: scale factor to multiply after outer product of query and key. If None, it's set to
                          1 / sqrt(dim_feature // num_heads)
        :param hierarchical: Whether to use hierarchical structure or not.
        :param num_joints: Number of joints.
        :param use_temporal_similarity: If true, for temporal GCN uses top-k similarity between nodes
        :param temporal_connection_len: Connects joint to itself within next `temporal_connection_len` frames
        :param use_tcn: If true, uses MS-TCN for temporal part of the graph branch.
        :param graph_only: Uses GCN instead of GraphFormer in the graph branch.
        :param neighbour_num: Number of neighbors for temporal GCN similarity.
        :param n_frames: Number of frames. Default is 243
        """
        super().__init__()

        self.joints_embed = nn.Linear(dim_in, dim_feat)
        self.quat_embed = nn.Linear(1, dim_feat)
        self.quat_linear = nn.Linear(dim_out, 1)
        self.trans_embed = nn.Linear(1, dim_feat)
        self.trans_linear = nn.Linear(dim_out, 1)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_joints, dim_feat))
        self.type_embed = nn.Parameter(torch.zeros(3, 1, 1, dim_feat))
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(timestep_embed_dim),
            nn.Linear(timestep_embed_dim, timestep_embed_dim * 2),
            nn.GELU(),
            nn.Linear(timestep_embed_dim * 2, dim_feat),
        )
        self.norm = nn.LayerNorm(dim_feat)
        act_layer = getattr(nn, act_layer) if isinstance(act_layer, str) else act_layer

        self.layers = create_layers(
            dim=dim_feat,
            n_layers=n_layers,
            mlp_ratio=mlp_ratio,
            act_layer=act_layer,
            attn_drop=attn_drop,
            drop_rate=drop,
            drop_path_rate=drop_path,
            num_heads=num_heads,
            use_layer_scale=use_layer_scale,
            qkv_bias=qkv_bias,
            qkv_scale=qkv_scale,
            layer_scale_init_value=layer_scale_init_value,
            use_adaptive_fusion=use_adaptive_fusion,
            hierarchical=hierarchical,
            use_temporal_similarity=use_temporal_similarity,
            temporal_connection_len=temporal_connection_len,
            use_tcn=use_tcn,
            graph_only=graph_only,
            neighbour_num=neighbour_num,
            n_frames=n_frames,
        )

        self.rep_logit = nn.Sequential(OrderedDict([("fc", nn.Linear(dim_feat, dim_rep)), ("act", nn.Tanh())]))

        self.head = nn.Linear(dim_rep, dim_out)

    def forward(
        self,
        x: torch.Tensor,
        quat: torch.Tensor,
        trans: torch.Tensor,
        t: torch.Tensor | None = None,
        return_rep: bool = False,
    ):
        """Forward pass of the MotionAGFormer model.

        Args:
            x (torch.Tensor): Input 2D from two-views (B, T, J, C=3*2).
            quat (torch.Tensor): Quaternions for cam1 to cam2 (B, 4).
            trans (torch.Tensor): Translation vector for cam1 to cam2 (B, 3).
            t (torch.Tensor): Time step embedding.  # NOTE: We need to decide the shape.
            return_rep (bool): If True, returns the representation logits instead of the final output.

        Returns:
            pred_pose (torch.Tensor): Predicted 2D poses from 2 views (B, T, J, 3*2).
            pred_quat (torch.Tensor): Predicted quaternion (B, 4).
            pred_trans (torch.Tensor): Predicted translation (B, 3).

        """
        quat = quat.unsqueeze(-1)  # (B, 4. 1)
        trans = trans.unsqueeze(-1)  # (B, 3, 1)

        bs, frames, joints, _ = x.shape
        x = self.joints_embed(x)  # (B, T, J, D)
        x = x + self.pos_embed + self.type_embed[0]

        quat = self.quat_embed(quat).unsqueeze(2)  # (B, 4, 1, D)
        quat = quat.expand(-1, -1, joints, -1) + self.pos_embed + self.type_embed[1]  # (B, 4, J, D)
        trans = self.trans_embed(trans).unsqueeze(2)  # (B, 3, 1, D)
        trans = trans.expand(-1, -1, joints, -1) + self.pos_embed + self.type_embed[2]  # (B, 3, J, D)
        x = torch.cat((x, quat, trans), dim=1)  # (B, T+7, J, D)

        if t is not None:
            time_embed = self.time_mlp(t)  # (B, 1, D)
            time_embed = time_embed.unsqueeze(1).unsqueeze(1)  # (B, 1, 1, D)
            time_embed = time_embed.expand(bs, x.size(1), x.size(2), -1)  # (B, T+7, J, D)
            x = x + time_embed

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        x = self.rep_logit(x)
        if return_rep:
            return x

        x = self.head(x)  # (B, T+7, J, D_out)
        pred_pose = x[:, :frames, :, :]  # (B, T, J, D_out)

        pred_quat = x[:, frames : frames + 4, :, :].mean(dim=2)  # (B, 4, D_out)
        pred_quat = self.quat_linear(pred_quat)  # (B, 4, 1)
        pred_quat = F.normalize(pred_quat, dim=1)
        mask = pred_quat[..., 0] < 0
        pred_quat[mask] = -pred_quat[mask]

        pred_trans = x[:, frames + 4 :, :, :].mean(dim=2)  # (B, 3, D_out)
        pred_trans = self.trans_linear(pred_trans)  # (B, 3, 1)
        pred_trans = F.normalize(pred_trans, dim=1)

        pred_quat = pred_quat.squeeze(-1)  # (B, 4)
        pred_trans = pred_trans.squeeze(-1)  # (B, 3)

        return pred_pose, pred_quat, pred_trans
