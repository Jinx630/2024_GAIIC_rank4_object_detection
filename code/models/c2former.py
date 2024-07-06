# -!- coding: utf-8 -!-
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from torch.nn import init

class Indexer(nn.Module):
    """获取C2former融合后的RGB或者TIR特征"""
    def __init__(self, c1, index):
        super().__init__()
        self.index = index

    def forward(self, x):
        if self.index == 0:
            return x[0]
        elif self.index == 1:
            return x[1]
        # return torch.add(x[0], x[1])


class C2Former(nn.Module):

    def __init__(
            self, dims_in,  n_heads, n_head_channels, n_groups, stage_idx, dims_out, q_size=(128, 160), kv_size=(128, 160),
            attn_drop=0.0, proj_drop=0.0, stride=3,
            offset_range_factor=2,
            no_off=False
    ):
        '''

        :param q_size:
        :param kv_size:
        :param n_heads: [6, 12, 24] 三层融合，每层对应值
        :param n_head_channels: dims_out[i] // num_heads [192/6, 384/12, 384/24]
        :param n_groups: [1, 2, 3]
        :param attn_drop:
        :param proj_drop:
        :param stride:
        :param offset_range_factor:
        :param no_off:
        :param stage_idx:[1, 2, 3]
        :param dims_in: 输入维度和输出维度 参考值 [512, 1024, 1024]
        :param dims_out: 中间层缩放维度 参考值 [192, 384, 384]
        '''

        super(C2Former, self).__init__()
        self.n_head_channels = n_head_channels
        self.scale = self.n_head_channels ** -0.5
        self.n_heads = n_heads
        self.q_h, self.q_w = q_size
        self.kv_h, self.kv_w = kv_size
        self.nc = n_head_channels * n_heads
        self.qnc = n_head_channels * n_heads * 2
        self.n_groups = n_groups
        self.n_group_channels = self.nc // self.n_groups
        self.n_group_heads = self.n_heads // self.n_groups
        self.no_off = no_off
        self.offset_range_factor = offset_range_factor

        ksizes = [9, 7, 5, 3]
        kk = ksizes[stage_idx]
        self.visinputconv = nn.Sequential(nn.Conv2d(dims_in, dims_out, (1, 1), (1, 1)), nn.ReLU())
        self.lwirinputconv = nn.Sequential(nn.Conv2d(dims_in, dims_out, (1, 1), (1, 1)), nn.ReLU())
        self.visoutputconv = nn.Sequential(nn.Conv2d(dims_out, dims_in, (1, 1), (1, 1)), nn.ReLU())
        self.lwiroutputconv = nn.Sequential(nn.Conv2d(dims_out, dims_in, (1, 1), (1, 1)), nn.ReLU())

        self.conv_offset = nn.Sequential(
            nn.Conv2d(self.n_group_channels, self.n_group_channels, kk, stride, kk // 2, groups=self.n_group_channels),
            LayernormProxy(self.n_group_channels),
            nn.GELU(),
            nn.Conv2d(self.n_group_channels, 2, 1, 1, 0, bias=False)
        )

        self.proj_q_lwir = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )
        self.proj_q_vis = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )
        self.proj_combinq = nn.Conv2d(
            self.qnc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_k_lwir = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )
        self.proj_k_vis = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )
        self.proj_v_lwir = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )
        self.proj_v_vis = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_out_lwir = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )
        self.proj_out_vis = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )
        self.vis_proj_drop = nn.Dropout(proj_drop, inplace=True)
        self.lwir_proj_drop = nn.Dropout(proj_drop, inplace=True)
        self.vis_attn_drop = nn.Dropout(attn_drop, inplace=True)
        self.lwir_attn_drop = nn.Dropout(attn_drop, inplace=True)

        self.vis_MN = Modalitynorm(self.nc, use_residual=True, learnable=True)
        self.lwir_MN = Modalitynorm(self.nc, use_residual=True, learnable=True)

        self.apply(self._init_weights)
    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Conv2d):
            init.kaiming_normal_(module.weight, mode="fan_out")
            if module.bias is not None:
                init.constant_(module.bias, 0)
        elif isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    @torch.no_grad()
    def _get_ref_points(self, H_key, W_key, B, dtype, device):

        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_key - 0.5, H_key, dtype=dtype, device=device),
            torch.linspace(0.5, W_key - 0.5, W_key, dtype=dtype, device=device)
        )
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(W_key).mul_(2).sub_(1)
        ref[..., 0].div_(H_key).mul_(2).sub_(1)
        ref = ref[None, ...].expand(B * self.n_groups, -1, -1, -1)  # B * g H W 2

        return ref

    def forward(self, x):
        vis_x_, lwir_x_ = x[0], x[1]
        vis_x, lwir_x = x[0], x[1]
        vis_x = self.visinputconv(vis_x)
        lwir_x = self.lwirinputconv(lwir_x)

        B, C, H, W = vis_x.size()
        dtype, device = vis_x.dtype, vis_x.device
        # concat two tensor
        x = torch.cat([vis_x, lwir_x], 1)
        combin_q = self.proj_combinq(x)

        q_off = einops.rearrange(combin_q, 'b (g c) h w -> (b g) c h w', g=self.n_groups, c=self.n_group_channels).contiguous()
        offset = self.conv_offset(q_off)
        Hk, Wk = offset.size(2), offset.size(3)
        n_sample = Hk * Wk

        if self.offset_range_factor > 0:
            offset_range = torch.tensor([1.0 / Hk, 1.0 / Wk], device=device).reshape(1, 2, 1, 1)
            offset = offset.tanh().mul(offset_range).mul(self.offset_range_factor)

        offset = einops.rearrange(offset, 'b p h w -> b h w p').contiguous()
        vis_reference = self._get_ref_points(Hk, Wk, B, dtype, device)
        lwir_reference = self._get_ref_points(Hk, Wk, B, dtype, device)

        if self.no_off:
            offset = offset.fill(0.0)

        if self.offset_range_factor >= 0:
            vis_pos = vis_reference + offset
            lwir_pos = lwir_reference
        else:
            vis_pos = (vis_reference + offset).tanh()
            lwir_pos = lwir_reference.tanh()
        if vis_x.dtype == torch.float16:
            vis_pos = vis_pos.half()
            lwir_pos = lwir_pos.half()
        vis_x_sampled = F.grid_sample(
            input=vis_x.reshape(B * self.n_groups, self.n_group_channels, H, W),
            grid=vis_pos[..., (1, 0)],
            mode='bilinear', align_corners=True)

        lwir_x_sampled = F.grid_sample(
            input=lwir_x.reshape(B * self.n_groups, self.n_group_channels, H, W),
            grid=lwir_pos[..., (1, 0)],
            mode='bilinear', align_corners=True)

        vis_x_sampled = vis_x_sampled.reshape(B, C, 1, n_sample)
        lwir_x_sampled = lwir_x_sampled.reshape(B, C, 1, n_sample)

        q_lwir = self.proj_q_lwir(self.vis_MN(vis_x, lwir_x))
        q_lwir = q_lwir.reshape(B * self.n_heads, self.n_head_channels, H * W)
        k_vis = self.proj_k_vis(vis_x_sampled).reshape(B * self.n_heads, self.n_head_channels, n_sample)
        v_vis = self.proj_v_vis(vis_x_sampled).reshape(B * self.n_heads, self.n_head_channels, n_sample)

        q_vis = self.proj_q_vis(self.lwir_MN(lwir_x, vis_x))
        q_vis = q_vis.reshape(B * self.n_heads, self.n_head_channels, H * W)
        k_lwir = self.proj_k_lwir(lwir_x_sampled).reshape(B * self.n_heads, self.n_head_channels, n_sample)
        v_lwir = self.proj_v_lwir(lwir_x_sampled).reshape(B * self.n_heads, self.n_head_channels, n_sample)

        attn_vis = torch.einsum('b c m, b c n -> b m n', q_lwir, k_vis).contiguous()
        attn_vis = attn_vis.mul(self.scale)
        attn_vis = F.softmax(attn_vis, dim=2)
        attn_vis = self.vis_attn_drop(attn_vis)
        out_vis = torch.einsum('b m n, b c n -> b c m', attn_vis, v_vis).contiguous()
        out_vis = out_vis.reshape(B, C, H, W)
        out_vis = self.vis_proj_drop(self.proj_out_vis(out_vis))

        attn_lwir = torch.einsum('b c m, b c n -> b m n', q_vis, k_lwir).contiguous()
        attn_lwir = attn_lwir.mul(self.scale)
        attn_lwir = F.softmax(attn_lwir, dim=2)
        attn_lwir = self.lwir_attn_drop(attn_lwir)
        out_lwir = torch.einsum('b m n, b c n -> b c m', attn_lwir, v_lwir).contiguous()
        out_lwir = out_lwir.reshape(B, C, H, W)
        out_lwir = self.lwir_proj_drop(self.proj_out_lwir(out_lwir))

        out_vis_ = self.visoutputconv(out_vis)
        out_lwir_ = self.lwiroutputconv(out_lwir)

        out_lwir = out_vis_+lwir_x_
        out_vis = out_lwir_+vis_x_
        return out_vis, out_lwir


class LayernormProxy(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = einops.rearrange(x, 'b c h w -> b h w c').contiguous()
        x = self.norm(x)
        return einops.rearrange(x, 'b h w c -> b c h w').contiguous()


# Modality Norm
class Modalitynorm(nn.Module):
    def __init__(self, nf, use_residual=True, learnable=True):
        super(Modalitynorm, self).__init__()

        self.learnable = learnable
        self.norm_layer = nn.InstanceNorm2d(nf, affine=False)

        if self.learnable:
            self.conv = nn.Sequential(nn.Conv2d(nf, nf, 3, 1, 1, bias=True),
                                      nn.ReLU(inplace=True))
            self.conv_gamma = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
            self.conv_beta = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

            self.use_residual = use_residual

            # initialization
            self.conv_gamma.weight.data.zero_()
            self.conv_beta.weight.data.zero_()
            self.conv_gamma.bias.data.zero_()
            self.conv_beta.bias.data.zero_()

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

    def forward(self, lr, ref):
        ref_normed = self.norm_layer(ref)
        if self.learnable:
            x = self.conv(lr)
            gamma = self.conv_gamma(x)
            beta = self.conv_beta(x)

        b, c, h, w = lr.size()
        lr = lr.view(b, c, h * w).contiguous()
        lr_mean = torch.mean(lr, dim=-1, keepdim=True).unsqueeze(3)
        lr_std = torch.std(lr, dim=-1, keepdim=True).unsqueeze(3)

        if self.learnable:
            if self.use_residual:
                gamma = gamma + lr_std
                beta = beta + lr_mean
            else:
                gamma = 1 + gamma
        else:
            gamma = lr_std
            beta = lr_mean

        out = ref_normed * gamma + beta

        return out