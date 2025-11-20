import torch as th
import torch.nn as nn
import torch

from .nn import SiLU, conv_nd, linear, zero_module, timestep_embedding, normalization
from .residual_blocks import (
    ResBlock,
    Downsample,
    Upsample,
    TimestepEmbedSequential,
)
from .attention_blocks import AttentionBlock
from functools import partial


class RAWControlNet(nn.Module):

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        c_channels=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        mid_attention=True,
        conditional_block_name="RGBGuidedResidualBlock",
        norm_num_groups=8,
        use_film=False,
        cond_channels=None,
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        attention_resolutions_ds = []
        for res in attention_resolutions:
            attention_resolutions_ds.append(image_size // int(res))

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions_ds = attention_resolutions_ds
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.c_channels = c_channels
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

        self.normalization_fn = partial(
            normalization,
            num_groups=norm_num_groups,
        )

        self.use_film = use_film
        self.cond_channels = cond_channels
        if self.use_film and self.cond_channels is None:
            raise ValueError(
                "RAWControlNet: use_film=True but cond_channels is None. ")

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList([
            TimestepEmbedSequential(
                conv_nd(dims,
                        in_channels,
                        ch,
                        3,
                        padding=1,
                        padding_mode="reflect"))
        ])

        self.input_zero_convs = nn.ModuleList(
            [zero_module(conv_nd(dims, ch, ch, 1))])

        if self.use_film:
            from .residual_blocks import FiLMResidualBlock as BaseResBlock
        else:
            BaseResBlock = ResBlock

        base_resblock_kwargs = dict(
            emb_channels=time_embed_dim,
            dropout=dropout,
            dims=dims,
            use_checkpoint=use_checkpoint,
            use_scale_shift_norm=use_scale_shift_norm,
            normalization_fn=self.normalization_fn,
        )

        if self.use_film:
            base_resblock_kwargs["cond_channels"] = cond_channels

        resblock_standard = partial(BaseResBlock, **base_resblock_kwargs)

        if conditional_block_name == "RGBGuidedResidualBlock":
            from rawdiffusion.models.residual_blocks import RGBGuidedResidualBlock
            resblock_guidance_cls = partial(RGBGuidedResidualBlock)
            guidance_extra_kwargs = {}
        elif conditional_block_name == "ResBlock":
            resblock_guidance_cls = ResBlock
            guidance_extra_kwargs = {}
            if self.use_film:
                guidance_extra_kwargs["cond_channels"] = cond_channels
        else:
            raise ValueError(
                f"Unknown conditional block name: {conditional_block_name}")

        resblock_guidance = partial(
            resblock_guidance_cls,
            emb_channels=time_embed_dim,
            dropout=dropout,
            c_channels=c_channels,
            dims=dims,
            use_checkpoint=use_checkpoint,
            use_scale_shift_norm=use_scale_shift_norm,
            normalization_fn=self.normalization_fn,
            **guidance_extra_kwargs,
        )

        attention = partial(
            AttentionBlock,
            use_checkpoint=use_checkpoint,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            use_new_attention_order=use_new_attention_order,
            normalization_fn=self.normalization_fn,
        )

        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    resblock_standard(
                        ch,
                        out_channels=int(mult * model_channels),
                    )
                ]
                ch = int(mult * model_channels)
                if ds in self.attention_resolutions_ds:
                    layers.append(attention(ch, ))

                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self.input_zero_convs.append(
                    zero_module(conv_nd(dims, ch, ch, 1)))

                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        resblock_standard(
                            ch,
                            out_channels=out_ch,
                            down=True,
                        ) if resblock_updown else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch))
                )
                self.input_zero_convs.append(
                    zero_module(conv_nd(dims, out_ch, out_ch, 1)))

                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            resblock_guidance(ch, ),
            (attention(ch) if mid_attention else nn.Identity()),
            resblock_guidance(ch, ),
        )
        self.middle_zero_conv = zero_module(conv_nd(dims, ch, ch, 1))
        self._feature_size += ch

    def forward(self, x, timesteps, guidance_features, cond=None):
        """
        Args:
            x:                [B, in_channels, H, W] noisy RGB input (same as RAWDiffusion).
            timesteps:        [B] diffusion timesteps.
            guidance_features:[B, c_channels, H, W] output of frozen rgb_guidance_module.
                               (Already computed once and shared with RAWDiffusionModel.)
            cond:             [B, cond_channels] FiLM condition vector if use_film=True.
        """

        if self.use_film and cond is None:
            raise ValueError(
                "RAWControlNet is configured with use_film=True, but no cond was passed to forward()."
            )

        hs = []
        emb = self.time_embed(
            timestep_embedding(timesteps, self.model_channels))

        h = x.type(self.dtype)
        for block, zero_conv in zip(self.input_blocks, self.input_zero_convs):
            if self.use_film:
                h = block(h, guidance_features, emb, cond)
            else:
                h = block(h, guidance_features, emb)
            hs.append(zero_conv(h))

        if self.use_film:
            h = self.middle_block(h, guidance_features, emb, cond)
        else:
            h = self.middle_block(h, guidance_features, emb)

        out = self.middle_zero_conv(h)

        return out
