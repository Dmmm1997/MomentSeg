import torch
from torch import nn
from torch.nn import functional as F
import torch
import copy
import math


class PositionEmbeddingSine(nn.Module):
    """Sinusoidal position embedding used in DETR model.

    Please see `End-to-End Object Detection with Transformers
    <https://arxiv.org/pdf/2005.12872>`_ for more details.

    Args:
        num_pos_feats (int): The feature dimension for each position along
            x-axis or y-axis. The final returned dimension for each position
            is 2 times of the input value.
        temperature (int, optional): The temperature used for scaling
            the position embedding. Default: 10000.
        scale (float, optional): A scale factor that scales the position
            embedding. The scale will be used only when `normalize` is True.
            Default: 2*pi.
        eps (float, optional): A value added to the denominator for numerical
            stability. Default: 1e-6.
        offset (float): An offset added to embed when doing normalization.
        normalize (bool, optional): Whether to normalize the position embedding.
            Default: False.
    """

    def __init__(
        self,
        num_pos_feats: int = 64,
        temperature: int = 10000,
        scale: float = 2 * math.pi,
        eps: float = 1e-6,
        offset: float = 0.0,
        normalize: bool = False,
    ):
        super().__init__()
        if normalize:
            assert isinstance(scale, (float, int)), (
                "when normalize is set,"
                "scale should be provided and in float or int type, "
                f"found {type(scale)}"
            )
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale
        self.eps = eps
        self.offset = offset

    def forward(self, mask: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward function for `PositionEmbeddingSine`.

        Args:
            mask (torch.Tensor): ByteTensor mask. Non-zero values representing
                ignored positions, while zero values means valid positions
                for the input tensor. Shape as `(bs, h, w)`.

        Returns:
            torch.Tensor: Returned position embedding with
            shape `(bs, num_pos_feats * 2, h, w)`
        """
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            y_embed = (y_embed + self.offset) / (y_embed[:, -1:, :] + self.eps) * self.scale
            x_embed = (x_embed + self.offset) / (x_embed[:, :, -1:] + self.eps) * self.scale
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature ** (
            2 * torch.div(dim_t, 2, rounding_mode="floor") / self.num_pos_feats
        )
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t

        # use view as mmdet instead of flatten for dynamically exporting to ONNX
        B, H, W = mask.size()
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).view(
            B, H, W, -1
        )
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).view(
            B, H, W, -1
        )
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

class GlobalCrossAttention(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0.0, proj_drop=0.0, rpe_hidden_dim=512):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)
        self.score_mlp = self.build_cpb_mlp(1, rpe_hidden_dim, num_heads)

    def build_cpb_mlp(self, in_dim, hidden_dim, out_dim):
        cpb_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=True), nn.ReLU(inplace=True), nn.Linear(hidden_dim, out_dim, bias=False)
        )
        return cpb_mlp

    def forward(
        self,
        query,
        k_input_flatten,
        v_input_flatten,
        input_padding_mask=None,
        attn_mask=None,  # B,1,N
    ):

        B_, N, C = k_input_flatten.shape
        k = self.k(k_input_flatten).reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(v_input_flatten).reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        B_, N, C = query.shape
        q = self.q(query).reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        q = q * self.scale

        attn = q @ k.transpose(-2, -1)

        if attn_mask is not None:
            attn_mask = self.score_mlp(attn_mask.unsqueeze(-1)).permute(0, 3, 1, 2)  # (B,1,N,nhead)
            assert attn.shape == attn_mask.shape
            attn += attn_mask

        if input_padding_mask is not None:
            # attn += input_padding_mask[:, None, None] * -100
            input_mask = input_padding_mask.reshape(input_padding_mask.size(0), 1, 1, -1).repeat(
                1, self.num_heads, 1, 1
            )
            attn += input_mask * -100
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = attn @ v
        x = x.transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class GlobalDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model=256,
        d_ffn=1024,
        dropout=0.1,
        activation="relu",
        n_heads=8,
        norm_type="post_norm",
    ):
        super().__init__()

        self.norm_type = norm_type

        # global cross attention
        self.cross_attn = GlobalCrossAttention(d_model, n_heads)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_pre(
        self, tgt, query_pos, src, src_pos_embed, src_padding_mask=None, self_attn_mask=None, cross_attn_mask=None
    ):
        # self attention
        tgt2 = self.norm2(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(
            q.transpose(0, 1),
            k.transpose(0, 1),
            tgt2.transpose(0, 1),
            attn_mask=self_attn_mask,
        )[
            0
        ].transpose(0, 1)
        tgt = tgt + self.dropout2(tgt2)

        # global cross attention
        tgt2 = self.norm1(tgt)
        tgt2 = self.cross_attn(
            self.with_pos_embed(tgt2, query_pos),
            self.with_pos_embed(src, src_pos_embed),
            src,
            src_padding_mask,
            cross_attn_mask,
        )
        tgt = tgt + self.dropout1(tgt2)

        # ffn
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout4(tgt2)

        return tgt

    def forward_post(
        self, tgt, query_pos, src, src_pos_embed, src_padding_mask=None, self_attn_mask=None, cross_attn_mask=None
    ):
        # self attention
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(
            q.transpose(0, 1),
            k.transpose(0, 1),
            tgt.transpose(0, 1),
            attn_mask=self_attn_mask,
        )[
            0
        ].transpose(0, 1)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # cross attention
        tgt2 = self.cross_attn(
            self.with_pos_embed(tgt, query_pos),
            self.with_pos_embed(src, src_pos_embed),
            src,
            src_padding_mask,
            cross_attn_mask,
        )
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)

        return tgt

    def forward(
        self, tgt, query_pos, src, src_pos_embed, src_padding_mask=None, self_attn_mask=None, cross_attn_mask=None
    ):
        if self.norm_type == "pre_norm":
            return self.forward_pre(
                tgt, query_pos, src, src_pos_embed, src_padding_mask, self_attn_mask, cross_attn_mask
            )
        if self.norm_type == "post_norm":
            return self.forward_post(
                tgt, query_pos, src, src_pos_embed, src_padding_mask, self_attn_mask, cross_attn_mask
            )


class SEG_ATTN(nn.Module):
    def __init__(self, hidden_channels):
        super(SEG_ATTN, self).__init__()
        self.DETAttn = GlobalDecoderLayer(
            d_model=hidden_channels, d_ffn=1024, dropout=0.1, activation="gelu", n_heads=8
        )
        self.SegAttn = GlobalDecoderLayer(
            d_model=hidden_channels, d_ffn=1024, dropout=0.1, activation="gelu", n_heads=8
        )
        self.position_embedding = PositionEmbeddingSine(
            num_pos_feats=hidden_channels // 2,
            temperature=10000,
            normalize=True,
        )
        self.query_embed = nn.Embedding(1, hidden_channels)

    def x_mask_pos_enc(self, x, img_shape):
        batch_size = x.size(0)
        input_img_h, input_img_w = img_shape
        x_mask = x.new_ones((batch_size, input_img_h, input_img_w))
        for img_id in range(batch_size):
            img_h, img_w = img_shape
            x_mask[img_id, :img_h, :img_w] = 0
        x_mask = F.interpolate(x_mask.unsqueeze(1), size=x.size()[-2:]).to(torch.bool).squeeze(1)
        x_pos_embeds = self.position_embedding(x_mask)
        return x_mask, x_pos_embeds

    def forward(self, query_feat, image_feat):
        assert query_feat.shape[0] == image_feat.shape[0]
        batch_size, hidden_size = image_feat.size(0), image_feat.size(1)
        query = self.query_embed.weight.unsqueeze(0).repeat(query_feat.shape[0], 1, 1)
        query_embed = self.DETAttn(
            tgt=query,
            query_pos=None,
            src=query_feat.unsqueeze(1),
            src_pos_embed=None,
            src_padding_mask=None,
            self_attn_mask=None,
            cross_attn_mask=None,
        )
        _, pos_embed = self.x_mask_pos_enc(image_feat, image_feat.shape[-2:])
        pos_embed = pos_embed.flatten(2).permute(0, 2, 1)
        seg_embed = self.SegAttn(
            tgt=query_embed,
            query_pos=None,
            src=image_feat.permute(0,2,3,1).reshape(batch_size, -1, hidden_size),
            src_pos_embed=pos_embed.to(image_feat.dtype),
            src_padding_mask=None,
            self_attn_mask=None,
            cross_attn_mask=None,
        )

        seg_embed = query_feat + seg_embed.squeeze(1) # N, 256

        return seg_embed