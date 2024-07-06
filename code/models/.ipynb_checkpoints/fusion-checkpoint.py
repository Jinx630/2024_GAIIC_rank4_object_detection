"""
@Time: 2024/4/27 19:57
@Author: xujinlingbj
@File: fusion.py
"""
import math
import sys

import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import torch
from torch.nn import init


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        # print(x.shape)
        return torch.cat(x, self.d)


class LearnableCoefficient(nn.Module):
    def __init__(self):
        super(LearnableCoefficient, self).__init__()
        self.bias = nn.Parameter(torch.FloatTensor([1.0]), requires_grad=True)

    def forward(self, x):
        out = x * self.bias
        return out


class SE_Block(nn.Module):
    def __init__(self, inchannel, ratio=16):
        super(SE_Block, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(inchannel, inchannel // ratio, bias=False),  # 从 c -> c/r
            nn.ReLU(),
            nn.Linear(inchannel // ratio, inchannel, bias=False),  # 从 c/r -> c
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, h, w = x.size()
        y = self.gap(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)

        return x * y.expand_as(x)

class SelfAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, h, attn_pdrop=0.1, resid_pdrop=0.1, vit_param = [], d_vit=512):
        """
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        """
        super(SelfAttention, self).__init__()
        assert d_k % h == 0
        self.d_model = d_model
        self.d_k = d_model // h
        self.d_v = d_model // h
        self.h = h

        # key, query, value projections for all heads
        self.que_proj_vis = nn.Linear(d_model, h * self.d_k)  # query projection
        self.key_proj_vis = nn.Linear(d_model, h * self.d_k)  # key projection
        self.val_proj_vis = nn.Linear(d_model, h * self.d_v)  # value projection

        self.que_proj_ir = nn.Linear(d_model, h * self.d_k)  # query projection
        self.key_proj_ir = nn.Linear(d_model, h * self.d_k)  # key projection
        self.val_proj_ir = nn.Linear(d_model, h * self.d_v)  # value projection

        self.out_proj_vis = nn.Linear(h * self.d_v, d_model)  # output projection
        self.out_proj_ir = nn.Linear(h * self.d_v, d_model)  # output projection

        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)

        # layer norm
        self.LN1 = nn.LayerNorm(d_model)
        self.LN2 = nn.LayerNorm(d_model)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x, attention_mask=None, attention_weights=None):
        """
        Computes Self-Attention
        Args:
            x (tensor): input (token) dim:(b_s, nx, c),
                b_s means batch size
                nx means length, for CNN, equals H*W, i.e. the length of feature maps
                c means channel, i.e. the channel of feature maps
            attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
            attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        Return:
            output (tensor): dim:(b_s, nx, c)
        """
        rgb_fea_flat = x[0]
        ir_fea_flat = x[1]
        b_s, nq = rgb_fea_flat.shape[:2]
        nk = rgb_fea_flat.shape[1]

        # Self-Attention
        rgb_fea_flat = self.LN1(rgb_fea_flat)
        q_vis = (
            self.que_proj_vis(rgb_fea_flat).contiguous().view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3).contiguous()
        )  # (b_s, h, nq, d_k)
        k_vis = (
            self.key_proj_vis(rgb_fea_flat).contiguous().view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1).contiguous()
        )  # (b_s, h, d_k, nk) K^T
        v_vis = (
            self.val_proj_vis(rgb_fea_flat).contiguous().view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3).contiguous()
        )  # (b_s, h, nk, d_v)

        ir_fea_flat = self.LN2(ir_fea_flat)
        q_ir = (
            self.que_proj_ir(ir_fea_flat).contiguous().view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3).contiguous()
        )  # (b_s, h, nq, d_k)
        k_ir = (
            self.key_proj_ir(ir_fea_flat).contiguous().view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1).contiguous()
        )  # (b_s, h, d_k, nk) K^T
        v_ir = (
            self.val_proj_ir(ir_fea_flat).contiguous().view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3).contiguous()
        )  # (b_s, h, nk, d_v)

        att_vis = torch.matmul(q_vis, k_vis) / np.sqrt(self.d_k)
        att_ir = torch.matmul(q_ir, k_ir) / np.sqrt(self.d_k)
        # att_vis = torch.matmul(k_vis, q_ir) / np.sqrt(self.d_k)
        # att_ir = torch.matmul(k_ir, q_vis) / np.sqrt(self.d_k)

        # get attention matrix
        att_vis = torch.softmax(att_vis, -1)
        att_vis = self.attn_drop(att_vis)
        att_ir = torch.softmax(att_ir, -1)
        att_ir = self.attn_drop(att_ir)

        # output
        out_vis = (
            torch.matmul(att_vis, v_vis).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)
        )  # (b_s, nq, h*d_v)
        out_vis = self.resid_drop(self.out_proj_vis(out_vis))  # (b_s, nq, d_model)
        out_ir = (
            torch.matmul(att_ir, v_ir).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)
        )  # (b_s, nq, h*d_v)
        out_ir = self.resid_drop(self.out_proj_ir(out_ir))  # (b_s, nq, d_model)

        return [out_vis, out_ir]

class CrossAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, h, attn_pdrop=0.1, resid_pdrop=0.1, vit_param = [], d_vit=512):
        """
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        """
        super(CrossAttention, self).__init__()
        assert d_k % h == 0
        self.d_model = d_model
        self.d_k = d_model // h
        self.d_v = d_model // h
        self.h = h

        # key, query, value projections for all heads
        self.que_proj_vis = nn.Linear(d_model, h * self.d_k)  # query projection
        self.key_proj_vis = nn.Linear(d_model, h * self.d_k)  # key projection
        self.val_proj_vis = nn.Linear(d_model, h * self.d_v)  # value projection

        self.que_proj_ir = nn.Linear(d_model, h * self.d_k)  # query projection
        self.key_proj_ir = nn.Linear(d_model, h * self.d_k)  # key projection
        self.val_proj_ir = nn.Linear(d_model, h * self.d_v)  # value projection

        self.out_proj_vis = nn.Linear(h * self.d_v, d_model)  # output projection
        self.out_proj_ir = nn.Linear(h * self.d_v, d_model)  # output projection

        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)

        # layer norm
        self.LN1 = nn.LayerNorm(d_model)
        self.LN2 = nn.LayerNorm(d_model)

        self.init_weights()

        if vit_param:

            self.que_proj_vis.weight.data[:d_vit,:d_vit] = vit_param[0]
            self.key_proj_vis.weight.data[:d_vit,:d_vit] = vit_param[1]
            self.val_proj_vis.weight.data[:d_vit,:d_vit] = vit_param[2]

            self.que_proj_ir.weight.data[:d_vit,:d_vit] = vit_param[0]
            self.key_proj_ir.weight.data[:d_vit,:d_vit] = vit_param[1]
            self.val_proj_ir.weight.data[:d_vit,:d_vit] = vit_param[2]

            self.que_proj_vis.bias.data[:d_vit] = vit_param[3]
            self.key_proj_vis.bias.data[:d_vit] = vit_param[4]
            self.val_proj_vis.bias.data[:d_vit] = vit_param[5]

            self.que_proj_vis.bias.data[:d_vit] = vit_param[3]
            self.key_proj_ir.bias.data[:d_vit] = vit_param[4]
            self.val_proj_ir.bias.data[:d_vit] = vit_param[5]

            self.out_proj_vis.weight.data[:d_vit,:d_vit] = vit_param[6]
            self.out_proj_ir.weight.data[:d_vit,:d_vit] = vit_param[6]

            self.out_proj_vis.bias.data[:d_vit] = vit_param[7]
            self.out_proj_ir.bias.data[:d_vit] = vit_param[7]

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x, attention_mask=None, attention_weights=None):
        """
        Computes Self-Attention
        Args:
            x (tensor): input (token) dim:(b_s, nx, c),
                b_s means batch size
                nx means length, for CNN, equals H*W, i.e. the length of feature maps
                c means channel, i.e. the channel of feature maps
            attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
            attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        Return:
            output (tensor): dim:(b_s, nx, c)
        """
        rgb_fea_flat = x[0]
        ir_fea_flat = x[1]
        b_s, nq = rgb_fea_flat.shape[:2]
        nk = rgb_fea_flat.shape[1]

        # Self-Attention
        rgb_fea_flat = self.LN1(rgb_fea_flat)
        q_vis = (
            self.que_proj_vis(rgb_fea_flat).contiguous().view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3).contiguous()
        )  # (b_s, h, nq, d_k)
        k_vis = (
            self.key_proj_vis(rgb_fea_flat).contiguous().view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1).contiguous()
        )  # (b_s, h, d_k, nk) K^T
        v_vis = (
            self.val_proj_vis(rgb_fea_flat).contiguous().view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3).contiguous()
        )  # (b_s, h, nk, d_v)

        ir_fea_flat = self.LN2(ir_fea_flat)
        q_ir = (
            self.que_proj_ir(ir_fea_flat).contiguous().view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3).contiguous()
        )  # (b_s, h, nq, d_k)
        k_ir = (
            self.key_proj_ir(ir_fea_flat).contiguous().view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1).contiguous()
        )  # (b_s, h, d_k, nk) K^T
        v_ir = (
            self.val_proj_ir(ir_fea_flat).contiguous().view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3).contiguous()
        )  # (b_s, h, nk, d_v)

        att_vis = torch.matmul(q_ir, k_vis) / np.sqrt(self.d_k)
        att_ir = torch.matmul(q_vis, k_ir) / np.sqrt(self.d_k)
        # att_vis = torch.matmul(k_vis, q_ir) / np.sqrt(self.d_k)
        # att_ir = torch.matmul(k_ir, q_vis) / np.sqrt(self.d_k)

        # get attention matrix
        att_vis = torch.softmax(att_vis, -1)
        att_vis = self.attn_drop(att_vis)
        att_ir = torch.softmax(att_ir, -1)
        att_ir = self.attn_drop(att_ir)

        # output
        out_vis = (
            torch.matmul(att_vis, v_vis).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)
        )  # (b_s, nq, h*d_v)
        out_vis = self.resid_drop(self.out_proj_vis(out_vis))  # (b_s, nq, d_model)
        out_ir = (
            torch.matmul(att_ir, v_ir).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)
        )  # (b_s, nq, h*d_v)
        out_ir = self.resid_drop(self.out_proj_ir(out_ir))  # (b_s, nq, d_model)

        return [out_vis, out_ir]


class CrossTransformerBlock(nn.Module):
    def __init__(self, d_model, d_k, d_v, h, block_exp, attn_pdrop, resid_pdrop, loops_num=1, vit_layer = 9):
        """
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        :param block_exp: Expansion factor for MLP (feed foreword network)
        """
        super(CrossTransformerBlock, self).__init__()
        self.loops = loops_num

        load_vit = False
        p = []
        d_vit = 512
        if vit_layer not in [0,1,2,3,4,5]:
            d_vit = 768
        if load_vit and vit_layer>0:
            checkpoint = torch.load("/mnt/workspace/workgroup/jinmu/ckpts/RS5M_VIT-B-32.pt", map_location="cpu")

            q,k,v = torch.chunk(checkpoint[f'visual.transformer.resblocks.{vit_layer}.attn.in_proj_weight'],3,dim=0)
            q,k,v = q[:d_vit,:d_vit],k[:d_vit,:d_vit],v[:d_vit,:d_vit]

            q_b,k_b,v_b = torch.chunk(checkpoint[f'visual.transformer.resblocks.{vit_layer}.attn.in_proj_bias'],3,dim=0)
            q_b,k_b,v_b = q_b[:d_vit],k_b[:d_vit],v_b[:d_vit]

            o = checkpoint[f'visual.transformer.resblocks.{vit_layer}.attn.out_proj.weight'][:d_vit,:d_vit]
            o_b = checkpoint[f'visual.transformer.resblocks.{vit_layer}.attn.out_proj.bias'][:d_vit]

            p = [q,k,v,q_b,k_b,v_b,o,o_b]
        
        self.selfatt = SelfAttention(d_model, d_k, d_v, h, attn_pdrop, resid_pdrop, p, d_vit=d_vit)
        self.crossatt = CrossAttention(d_model, d_k, d_v, h, attn_pdrop, resid_pdrop, p, d_vit=d_vit)
        self.mlp_vis = nn.Sequential(
            nn.Linear(d_model, block_exp * d_model),
            nn.GELU(),
            nn.Linear(block_exp * d_model, d_model),
            nn.Dropout(resid_pdrop),
        )
        self.mlp_ir = nn.Sequential(
            nn.Linear(d_model, block_exp * d_model),
            nn.GELU(),
            nn.Linear(block_exp * d_model, d_model),
            nn.Dropout(resid_pdrop),
        )

        self.LN1 = nn.LayerNorm(d_model)
        self.LN2 = nn.LayerNorm(d_model)

        self.coefficient1 = LearnableCoefficient()
        self.coefficient2 = LearnableCoefficient()
        self.coefficient3 = LearnableCoefficient()
        self.coefficient4 = LearnableCoefficient()
        self.coefficient5 = LearnableCoefficient()
        self.coefficient6 = LearnableCoefficient()
        self.coefficient7 = LearnableCoefficient()
        self.coefficient8 = LearnableCoefficient()

        if load_vit and vit_layer>0:

            self.LN1.weight.data[:d_vit] = checkpoint[f'visual.transformer.resblocks.{vit_layer}.ln_2.weight'][:d_vit]
            self.LN2.bias.data[:d_vit] = checkpoint[f'visual.transformer.resblocks.{vit_layer}.ln_2.bias'][:d_vit]

            self.mlp_vis[0].weight.data[:d_vit*4,:d_vit] = checkpoint[f'visual.transformer.resblocks.{vit_layer}.mlp.c_fc.weight'][:d_vit*4,:d_vit]
            self.mlp_vis[0].bias.data[:d_vit*4] = checkpoint[f'visual.transformer.resblocks.{vit_layer}.mlp.c_fc.bias'][:d_vit*4]
            self.mlp_vis[2].weight.data[:d_vit,:d_vit*4] = checkpoint[f'visual.transformer.resblocks.{vit_layer}.mlp.c_proj.weight'][:d_vit,:d_vit*4]
            self.mlp_vis[2].bias.data[:d_vit] = checkpoint[f'visual.transformer.resblocks.{vit_layer}.mlp.c_proj.bias'][:d_vit]

            self.mlp_ir[0].weight.data[:d_vit*4,:d_vit] = checkpoint[f'visual.transformer.resblocks.{vit_layer}.mlp.c_fc.weight'][:d_vit*4,:d_vit]
            self.mlp_ir[0].bias.data[:d_vit*4] = checkpoint[f'visual.transformer.resblocks.{vit_layer}.mlp.c_fc.bias'][:d_vit*4]
            self.mlp_ir[2].weight.data[:d_vit,:d_vit*4] = checkpoint[f'visual.transformer.resblocks.{vit_layer}.mlp.c_proj.weight'][:d_vit,:d_vit*4]
            self.mlp_ir[2].bias.data[:d_vit] = checkpoint[f'visual.transformer.resblocks.{vit_layer}.mlp.c_proj.bias'][:d_vit]

    def forward(self, x):
        rgb_fea_flat = x[0]
        ir_fea_flat = x[1]
        assert rgb_fea_flat.shape[0] == ir_fea_flat.shape[0]
        bs, nx, c = rgb_fea_flat.size()
        h = w = int(math.sqrt(nx))

        for loop in range(self.loops):
            try:
                rgb_fea_out, ir_fea_out = self.selfatt([rgb_fea_flat, ir_fea_flat])
            except:
                pass
            rgb_fea_out, ir_fea_out = self.crossatt([rgb_fea_flat, ir_fea_flat])
            
            rgb_att_out = self.coefficient1(rgb_fea_flat) + self.coefficient2(rgb_fea_out)
            ir_att_out = self.coefficient3(ir_fea_flat) + self.coefficient4(ir_fea_out)
            rgb_fea_flat = self.coefficient5(rgb_att_out) + self.coefficient6(self.mlp_vis(self.LN2(rgb_att_out)))
            ir_fea_flat = self.coefficient7(ir_att_out) + self.coefficient8(self.mlp_ir(self.LN2(ir_att_out)))

        return [rgb_fea_flat, ir_fea_flat]

class LearnableWeights(nn.Module):
    def __init__(self):
        super(LearnableWeights, self).__init__()
        self.w1 = nn.Parameter(torch.tensor([0.5]), requires_grad=True)
        self.w2 = nn.Parameter(torch.tensor([0.5]), requires_grad=True)

    def forward(self, x1, x2):
        out = x1 * self.w1 + x2 * self.w2
        return out

class AdaptivePool2d(nn.Module):
    def __init__(self, output_h, output_w, pool_type="avg"):
        super(AdaptivePool2d, self).__init__()

        self.output_h = output_h
        self.output_w = output_w
        self.pool_type = pool_type

    def forward(self, x):
        bs, c, input_h, input_w = x.shape

        if (input_h > self.output_h) or (input_w > self.output_w):
            self.stride_h = input_h // self.output_h
            self.stride_w = input_w // self.output_w
            self.kernel_size = (
                input_h - (self.output_h - 1) * self.stride_h,
                input_w - (self.output_w - 1) * self.stride_w,
            )

            if self.pool_type == "avg":
                y = nn.AvgPool2d(kernel_size=self.kernel_size, stride=(self.stride_h, self.stride_w), padding=0)(x)
            else:
                y = nn.MaxPool2d(kernel_size=self.kernel_size, stride=(self.stride_h, self.stride_w), padding=0)(x)
        else:
            y = x

        return y

#  ICAFusion 融合模块
class TransformerFusionBlock(nn.Module):
    def __init__(
        self,
        d_model,
        vert_anchors=16,
        horz_anchors=16,
        h=8,
        block_exp=4,
        n_layer=3,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        resid_pdrop=0.1,
    ):
        super(TransformerFusionBlock, self).__init__()
        print('n_lyer: ', n_layer)
        self.n_embd = d_model
        self.vert_anchors = vert_anchors
        self.horz_anchors = horz_anchors
        d_k = d_model
        d_v = d_model

        # positional embedding parameter (learnable), rgb_fea + ir_fea
        self.pos_emb_vis = nn.Parameter(torch.zeros(1, vert_anchors * horz_anchors, self.n_embd))
        self.pos_emb_ir = nn.Parameter(torch.zeros(1, vert_anchors * horz_anchors, self.n_embd))

        # downsampling
        # self.avgpool = nn.AdaptiveAvgPool2d((self.vert_anchors, self.horz_anchors))
        # self.maxpool = nn.AdaptiveMaxPool2d((self.vert_anchors, self.horz_anchors))
        # self.vert_anchors, self.horz_anchors 对应输出 h 和输出 w
        self.avgpool = AdaptivePool2d(self.vert_anchors, self.horz_anchors, "avg")
        self.maxpool = AdaptivePool2d(self.vert_anchors, self.horz_anchors, "max")

        # LearnableCoefficient
        self.vis_coefficient = LearnableWeights()
        self.ir_coefficient = LearnableWeights()

        # init weights
        self.apply(self._init_weights)

        vit_layer = -1
        if vert_anchors == 20:
            vit_layer = 0
        elif vert_anchors == 16:
            vit_layer = 6
        elif vert_anchors == 10:
            vit_layer = 6

        # cross transformer
        self.crosstransformer = nn.Sequential(
            *[
                CrossTransformerBlock(d_model, d_k, d_v, h, block_exp, attn_pdrop, resid_pdrop, vit_layer=vit_layer + layer)
                for layer in range(n_layer)
            ]
        )

        # Concat
        self.concat = Concat(dimension=1)

        # conv1x1
        self.conv1x1_out = Conv(c1=d_model * 2, c2=d_model, k=1, s=1, p=0, g=1, act=True)


    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x):
        rgb_fea = x[0]
        ir_fea = x[1]
        assert rgb_fea.shape[0] == ir_fea.shape[0]
        bs, c, h, w = rgb_fea.shape

        # ------------------------- cross-modal feature fusion -----------------------#
        # new_rgb_fea = (self.avgpool(rgb_fea) + self.maxpool(rgb_fea)) / 2
        # torch.Size([6, 512, 20, 20])
        new_rgb_fea = self.vis_coefficient(self.avgpool(rgb_fea), self.maxpool(rgb_fea))
        # print(new_rgb_fea.shape)
        # 512 20 20

        new_c, new_h, new_w = new_rgb_fea.shape[1], new_rgb_fea.shape[2], new_rgb_fea.shape[3]
        # torch.Size([6, 400, 512])
        rgb_fea_flat = new_rgb_fea.contiguous().view(bs, new_c, -1).permute(0, 2, 1).contiguous() + self.pos_emb_vis

        # print(rgb_fea_flat.shape)
        # print(new_c, new_h, new_w)
        # new_ir_fea = (self.avgpool(ir_fea) + self.maxpool(ir_fea)) / 2
        # torch.Size([6, 512, 20, 20])
        new_ir_fea = self.ir_coefficient(self.avgpool(ir_fea), self.maxpool(ir_fea))
        # print(new_ir_fea.shape)
        # torch.Size([6, 400, 512])

        ir_fea_flat = new_ir_fea.contiguous().view(bs, new_c, -1).permute(0, 2, 1).contiguous() + self.pos_emb_ir

        # print(ir_fea_flat.shape)
        # torch.Size([6, 400, 512]) torch.Size([6, 400, 512])
        rgb_fea_flat, ir_fea_flat = self.crosstransformer([rgb_fea_flat, ir_fea_flat])

        # print(rgb_fea_flat.shape, ir_fea_flat.shape)
        # torch.Size([6, 512, 20, 20])
        rgb_fea_CFE = rgb_fea_flat.contiguous().view(bs, new_h, new_w, new_c).permute(0, 3, 1, 2).contiguous()
        # print(rgb_fea_CFE.shape)
        # print(self.training) true
        if self.training == True:
            # torch.Size([6, 512, 80, 80])
            rgb_fea_CFE = F.interpolate(rgb_fea_CFE, size=([h, w]), mode="nearest")
        else:
            rgb_fea_CFE = F.interpolate(rgb_fea_CFE, size=([h, w]), mode="bilinear")
        # print(rgb_fea_CFE.shape)
        # torch.Size([6, 512, 80, 80])
        new_rgb_fea = rgb_fea_CFE + rgb_fea
        # print(new_rgb_fea.shape)
        # torch.Size([6, 512, 20, 20])
        ir_fea_CFE = ir_fea_flat.contiguous().view(bs, new_h, new_w, new_c).permute(0, 3, 1, 2).contiguous()
        # print(ir_fea_CFE.shape)
        if self.training == True:
            # torch.Size([6, 512, 80, 80])
            ir_fea_CFE = F.interpolate(ir_fea_CFE, size=([h, w]), mode="nearest")
        else:
            ir_fea_CFE = F.interpolate(ir_fea_CFE, size=([h, w]), mode="bilinear")
        # print(ir_fea_CFE.shape)
        # torch.Size([6, 512, 80, 80])
        new_ir_fea = ir_fea_CFE + ir_fea

        # print(new_ir_fea.shape)
        # torch.Size([6, 1024, 80, 80])
        new_fea = self.concat([new_rgb_fea, new_ir_fea])
        # print(new_fea.shape)
        # torch.Size([6, 512, 80, 80])
        new_fea = self.conv1x1_out(new_fea)
        # print(new_fea.shape)
        # sys.exit(0)
        # ------------------------- feature visulization -----------------------#
        # save_dir = '/home/shen/Chenyf/FLIR-align-3class/feature_save/'
        # fea_rgb = torch.mean(rgb_fea, dim=1)
        # fea_rgb_CFE = torch.mean(rgb_fea_CFE, dim=1)
        # fea_rgb_new = torch.mean(new_rgb_fea, dim=1)
        # fea_ir = torch.mean(ir_fea, dim=1)
        # fea_ir_CFE = torch.mean(ir_fea_CFE, dim=1)
        # fea_ir_new = torch.mean(new_ir_fea, dim=1)
        # fea_new = torch.mean(new_fea, dim=1)
        # block = [fea_rgb, fea_rgb_CFE, fea_rgb_new, fea_ir, fea_ir_CFE, fea_ir_new, fea_new]
        # black_name = ['fea_rgb', 'fea_rgb After CFE', 'fea_rgb skip', 'fea_ir', 'fea_ir After CFE', 'fea_ir skip', 'fea_ir NiNfusion']
        # plt.figure()
        # for i in range(len(block)):
        #     feature = transforms.ToPILImage()(block[i].squeeze())
        #     ax = plt.subplot(3, 3, i + 1)
        #     ax.set_xticks([])
        #     ax.set_yticks([])
        #     ax.set_title(black_name[i], fontsize=8)
        #     plt.imshow(feature)
        # plt.savefig(save_dir + 'fea_{}x{}.png'.format(h, w), dpi=300)
        # -----------------------------------------------------------------------------#

        return new_fea

