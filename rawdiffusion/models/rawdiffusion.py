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


class RAWDiffusionModel(nn.Module):

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        rgb_guidance_module,
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
        out_tanh=False,
        conditional_block_name="RGBGuidedResidualBlock",
        norm_num_groups=8,
        latent_drop_rate=0,
        use_film=False,
        film_cond_channels=None,
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        attention_resolutions_ds = []
        for res in attention_resolutions:
            attention_resolutions_ds.append(image_size // int(res))

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
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
        self.out_tanh = out_tanh

        self.normalization_fn = partial(normalization,
                                        num_groups=norm_num_groups)

        self.use_film = use_film
        self.film_cond_channels = film_cond_channels
        if self.use_film and self.film_cond_channels is None:
            raise ValueError(
                "RAWDiffusionModel: use_film=True but film_cond_channels is None. "
            )

        if rgb_guidance_module:
            self.rgb_guidance_module = rgb_guidance_module(
                normalization_fn=self.normalization_fn)
        else:
            self.rgb_guidance_module = None

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        ch = input_ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList([
            TimestepEmbedSequential(
                conv_nd(dims,
                        in_channels,
                        ch,
                        3,
                        padding=1,
                        padding_mode="reflect"))
        ])

        self.latent_drop_rate = latent_drop_rate

        if latent_drop_rate > 0:
            self.mask_token = th.nn.Parameter(th.randn(c_channels))
        else:
            self.mask_token = None

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
            base_resblock_kwargs["film_cond_channels"] = film_cond_channels

        resblock_standard = partial(BaseResBlock, **base_resblock_kwargs)

        if conditional_block_name == "RGBGuidedResidualBlock":
            from rawdiffusion.models.residual_blocks import RGBGuidedResidualBlock
            resblock_guidance_cls = partial(RGBGuidedResidualBlock)
            guidance_extra_kwargs = {}
        elif conditional_block_name == "ResBlock":
            resblock_guidance_cls = ResBlock
            guidance_extra_kwargs = {}
            if self.use_film:
                guidance_extra_kwargs[
                    "film_cond_channels"] = film_cond_channels
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

        attention_upsamle = partial(
            AttentionBlock,
            use_checkpoint=use_checkpoint,
            num_heads=num_heads_upsample,
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
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            resblock_guidance(ch, ),
            (attention(ch) if mid_attention else nn.Identity()),
            resblock_guidance(ch, ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    resblock_guidance(
                        ch + ich,
                        out_channels=int(model_channels * mult),
                    )
                ]
                ch = int(model_channels * mult)
                if ds in self.attention_resolutions_ds:
                    layers.append(attention_upsamle(ch, ))
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        resblock_guidance(
                            ch,
                            out_channels=out_ch,
                            up=True,
                        ) if resblock_updown else Upsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch))
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            self.normalization_fn(ch),
            SiLU(),
            zero_module(
                conv_nd(dims,
                        input_ch,
                        out_channels,
                        3,
                        padding=1,
                        padding_mode="reflect")),
        )

    def freeze_backbone(self):
        for p in self.parameters():
            p.requires_grad_(False)

    @staticmethod
    def make_camera_hint_from_film_cond(film_cond: torch.Tensor,
                                        x: torch.Tensor):
        if film_cond is None:
            return None

        film_cond = film_cond.to(device=x.device, dtype=x.dtype)
        B, _, H, W = x.shape
        return film_cond[..., None, None].expand(B, film_cond.shape[1], H, W)

    def forward(
        self,
        x,
        timesteps,
        guidance_data,
        film_cond=None,
        controlnet=None,
        controlnet_scale: float = 1.0,
    ):

        if self.use_film and film_cond is None:
            raise ValueError(
                "RAWDiffusionModel is configured with use_film=True, but no cond was passed to forward()."
            )

        if self.rgb_guidance_module is not None:
            guidance_features = self.rgb_guidance_module(guidance_data)

            if self.training and self.latent_drop_rate > 0:
                bs = guidance_features.shape[0]
                mask = (th.rand(bs, 1, 1, 1, device=guidance_features.device)
                        < self.latent_drop_rate).float()

                mask_token = self.mask_token[None, :, None, None]
                guidance_features = guidance_features * (
                    1 - mask) + mask_token * mask
        else:
            guidance_features = None

        control_input_res = None
        control_middle_res = None
        if controlnet is not None:
            control_out = controlnet(
                x=x,
                hint=self.make_camera_hint_from_film_cond(film_cond, x),
                timesteps=timesteps,
                guidance_features=guidance_features,
                film_cond=film_cond,
            )
            control_input_res = control_out["input_block_res"]
            control_middle_res = control_out["middle_block_res"]

        hs = []
        emb = self.time_embed(
            timestep_embedding(timesteps, self.model_channels))

        # Frozen Encoder
        h = x.type(self.dtype)
        for block in self.input_blocks:
            if self.use_film:
                h = block(h, guidance_features, emb, film_cond)
            else:
                h = block(h, guidance_features, emb)
            hs.append(h)

        # Bottleneck
        if self.use_film:
            h = self.middle_block(h, guidance_features, emb, film_cond)
        else:
            h = self.middle_block(h, guidance_features, emb)

        if control_middle_res is not None:
            h = h + controlnet_scale * control_middle_res

        # Decoder
        if control_input_res is not None:
            assert len(control_input_res) == len(hs)
            control_idx = len(control_input_res) - 1
        else:
            control_idx = None

        for block in self.output_blocks:
            skip = hs.pop()

            if control_idx is not None:
                skip = skip + controlnet_scale * control_input_res[control_idx]
                control_idx -= 1

            if h.shape[2:] != skip.shape[2:]:
                H = min(h.shape[2], skip.shape[2])
                W = min(h.shape[3], skip.shape[3])
                h = h[:, :, :H, :W]
                skip = skip[:, :, :H, :W]

            h = th.cat([h, skip], dim=1)

            if self.use_film:
                h = block(h, guidance_features, emb, film_cond)
            else:
                h = block(h, guidance_features, emb)

        h = h.type(x.dtype)
        out = self.out(h)

        if self.out_tanh:
            out = th.tanh(out)

        return out
