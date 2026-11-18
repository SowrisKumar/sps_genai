import math
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalEmbedding(nn.Module):
    def __init__(self, num_frequencies=16):
        super().__init__()

        self.num_frequencies = num_frequencies

        frequencies = torch.exp(
            torch.linspace(
                math.log(1.0),
                math.log(1000.0),
                num_frequencies
            )
        )

        self.register_buffer(
            "angular_speeds",
            2.0
            * math.pi
            * frequencies.view(1, 1, 1, -1)
        )

    def forward(self, x):

        x = x.expand(
            -1,
            1,
            1,
            self.num_frequencies
        )

        sin_part = torch.sin(
            self.angular_speeds * x
        )

        cos_part = torch.cos(
            self.angular_speeds * x
        )

        return torch.cat(
            [sin_part, cos_part],
            dim=-1
        )


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels
    ):
        super().__init__()

        if in_channels != out_channels:
            self.proj = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1
            )
        else:
            self.proj = nn.Identity()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1
        )

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1
        )

    def swish(self, x):
        return x * torch.sigmoid(x)

    def forward(self, x):

        residual = self.proj(x)

        x = self.swish(
            self.conv1(x)
        )

        x = self.conv2(x)

        return x + residual


class DownBlock(nn.Module):
    def __init__(
        self,
        width,
        block_depth,
        in_channels
    ):
        super().__init__()

        self.blocks = nn.ModuleList()

        for _ in range(block_depth):

            self.blocks.append(
                ResidualBlock(
                    in_channels,
                    width
                )
            )

            in_channels = width

        self.pool = nn.AvgPool2d(
            kernel_size=2
        )

    def forward(self, x, skips):

        for block in self.blocks:

            x = block(x)

            skips.append(x)

        x = self.pool(x)

        return x


class UpBlock(nn.Module):
    def __init__(
        self,
        width,
        block_depth,
        in_channels
    ):
        super().__init__()

        self.blocks = nn.ModuleList()

        for _ in range(block_depth):

            self.blocks.append(
                ResidualBlock(
                    in_channels + width,
                    width
                )
            )

            in_channels = width

    def forward(self, x, skips):

        x = F.interpolate(
            x,
            scale_factor=2,
            mode="bilinear",
            align_corners=False
        )

        for block in self.blocks:

            skip = skips.pop()

            x = torch.cat(
                [x, skip],
                dim=1
            )

            x = block(x)

        return x

class UNet(nn.Module):
    def __init__(
        self,
        image_size=64,
        num_channels=3,
        embedding_dim=32
    ):
        super().__init__()

        self.num_channels = num_channels
        self.image_size = image_size

        self.initial = nn.Conv2d(
            num_channels,
            32,
            kernel_size=1
        )

        self.embedding = SinusoidalEmbedding(
            num_frequencies=16
        )

        self.down1 = DownBlock(
            32,
            block_depth=2,
            in_channels=64
        )

        self.down2 = DownBlock(
            64,
            block_depth=2,
            in_channels=32
        )

        self.down3 = DownBlock(
            96,
            block_depth=2,
            in_channels=64
        )

        self.mid1 = ResidualBlock(
            96,
            128
        )

        self.mid2 = ResidualBlock(
            128,
            128
        )

        self.up1 = UpBlock(
            96,
            block_depth=2,
            in_channels=128
        )

        self.up2 = UpBlock(
            64,
            block_depth=2,
            in_channels=96
        )

        self.up3 = UpBlock(
            32,
            block_depth=2,
            in_channels=64
        )

        self.final = nn.Conv2d(
            32,
            num_channels,
            kernel_size=1
        )

        nn.init.zeros_(
            self.final.weight
        )

    def forward(
        self,
        noisy_images,
        noise_variances
    ):

        skips = []

        x = self.initial(
            noisy_images
        )

        noise_embedding = self.embedding(
            noise_variances
        )

        noise_embedding = (
            noise_embedding
            .permute(0, 3, 1, 2)
        )

        noise_embedding = F.interpolate(
            noise_embedding,
            size=(
                self.image_size,
                self.image_size
            ),
            mode="nearest"
        )

        x = torch.cat(
            [x, noise_embedding],
            dim=1
        )

        x = self.down1(x, skips)
        x = self.down2(x, skips)
        x = self.down3(x, skips)

        x = self.mid1(x)
        x = self.mid2(x)

        x = self.up1(x, skips)
        x = self.up2(x, skips)
        x = self.up3(x, skips)

        return self.final(x)
def offset_cosine_diffusion_schedule(
    diffusion_times,
    min_signal_rate=0.02,
    max_signal_rate=0.95
):
    start_angle = torch.acos(
        torch.tensor(
            max_signal_rate,
            dtype=torch.float32,
            device=diffusion_times.device
        )
    )

    end_angle = torch.acos(
        torch.tensor(
            min_signal_rate,
            dtype=torch.float32,
            device=diffusion_times.device
        )
    )

    diffusion_angles = (
        start_angle
        + diffusion_times * (end_angle - start_angle)
    )

    signal_rates = torch.cos(diffusion_angles)
    noise_rates = torch.sin(diffusion_angles)

    return noise_rates, signal_rates
class DiffusionModel(nn.Module):
    def __init__(self, network):
        super().__init__()

        self.network = network
        self.ema_network = copy.deepcopy(network)

        self.ema_network.eval()

        self.ema_decay = 0.8

        self.normalizer_mean = 0.0
        self.normalizer_std = 1.0

    def set_normalizer(self, mean, std):
        self.normalizer_mean = mean
        self.normalizer_std = std

    def denormalize(self, x):
        return torch.clamp(
            x * self.normalizer_std
            + self.normalizer_mean,
            0,
            1
        )

    def denoise(
        self,
        noisy_images,
        noise_rates,
        signal_rates,
        training=True
    ):
        if training:
            network = self.network
        else:
            network = self.ema_network

        predicted_noise = network(
            noisy_images,
            noise_rates ** 2
        )

        predicted_image = (
            noisy_images
            - noise_rates * predicted_noise
        ) / signal_rates

        return predicted_noise, predicted_image

    def train_step(
        self,
        images,
        optimizer,
        loss_fn
    ):
        images = (
            images - self.normalizer_mean
        ) / self.normalizer_std

        noise = torch.randn_like(images)

        diffusion_times = torch.rand(
            images.size(0),
            1,
            1,
            1,
            device=images.device
        )

        noise_rates, signal_rates = (
            offset_cosine_diffusion_schedule(
                diffusion_times
            )
        )

        noisy_images = (
            signal_rates * images
            + noise_rates * noise
        )

        predicted_noise, _ = self.denoise(
            noisy_images,
            noise_rates,
            signal_rates,
            training=True
        )

        loss = loss_fn(
            predicted_noise,
            noise
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        with torch.no_grad():

            for ema_param, param in zip(
                self.ema_network.parameters(),
                self.network.parameters()
            ):
                ema_param.copy_(
                    self.ema_decay * ema_param
                    + (1 - self.ema_decay) * param
                )

            for ema_buffer, buffer in zip(
                self.ema_network.buffers(),
                self.network.buffers()
            ):
                ema_buffer.copy_(buffer)

        return loss.item()

    def reverse_diffusion(
        self,
        initial_noise,
        diffusion_steps=20
    ):
        step_size = 1.0 / diffusion_steps

        current_images = initial_noise

        for step in range(diffusion_steps):

            diffusion_time = torch.ones(
                initial_noise.size(0),
                1,
                1,
                1,
                device=initial_noise.device
            ) * (
                1 - step * step_size
            )

            noise_rates, signal_rates = (
                offset_cosine_diffusion_schedule(
                    diffusion_time
                )
            )

            predicted_noise, predicted_image = (
                self.denoise(
                    current_images,
                    noise_rates,
                    signal_rates,
                    training=False
                )
            )

            next_time = (
                diffusion_time
                - step_size
            ).clamp(min=0)

            next_noise_rates, next_signal_rates = (
                offset_cosine_diffusion_schedule(
                    next_time
                )
            )

            current_images = (
                next_signal_rates
                * predicted_image
                + next_noise_rates
                * predicted_noise
            )

        return predicted_image

    def generate(
        self,
        num_images=1,
        diffusion_steps=20
    ):
        device = next(
            self.parameters()
        ).device

        noise = torch.randn(
            num_images,
            3,
            64,
            64,
            device=device
        )

        with torch.no_grad():

            images = self.reverse_diffusion(
                noise,
                diffusion_steps
            )

        return self.denormalize(images)
