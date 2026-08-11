from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer, TVAESynthesizer
from sklearn.preprocessing import StandardScaler

from src.config import RANDOM_STATE

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None

SUPPORTED_SYNTHESIZERS = ("CTGAN", "TVAE", "CUSTOM_GAN")

if nn is not None:

    class TabularGenerator(nn.Module):
        def __init__(
            self,
            noise_dim: int,
            output_dim: int,
            hidden_dims: Sequence[int] = (128, 128),
        ):
            super().__init__()

            layers: list[nn.Module] = []
            prev_dim = noise_dim

            for hidden_dim in hidden_dims:
                hidden_dim = int(hidden_dim)
                layers.append(nn.Linear(prev_dim, hidden_dim))
                layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.ReLU())
                prev_dim = hidden_dim

            layers.append(nn.Linear(prev_dim, output_dim))
            self.network = nn.Sequential(*layers)

        def forward(self, z):
            return self.network(z)

    class TabularDiscriminator(nn.Module):
        def __init__(
            self,
            input_dim: int,
            hidden_dims: Sequence[int] = (128, 64),
            dropout: float = 0.2,
        ):
            super().__init__()

            layers: list[nn.Module] = []
            prev_dim = input_dim

            for hidden_dim in hidden_dims:
                hidden_dim = int(hidden_dim)
                layers.append(nn.Linear(prev_dim, hidden_dim))
                layers.append(nn.LeakyReLU(0.2))
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
                prev_dim = hidden_dim

            layers.append(nn.Linear(prev_dim, 1))
            layers.append(nn.Sigmoid())
            self.network = nn.Sequential(*layers)

        def forward(self, x):
            return self.network(x).squeeze(-1)

else:

    class TabularGenerator:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for TabularGenerator. Install torch first.")


    class TabularDiscriminator:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for TabularDiscriminator. Install torch first.")


class CustomTabularGANSynthesizer:
    def __init__(
        self,
        *,
        noise_dim: int = 32,
        generator_hidden_dims: Sequence[int] = (128, 128),
        discriminator_hidden_dims: Sequence[int] = (128, 64),
        dropout: float = 0.2,
        epochs: int = 500,
        batch_size: int = 64,
        learning_rate: float = 2e-4,
        random_state: int = RANDOM_STATE,
        device: str | None = None,
        verbose: bool = True,
    ):
        self.noise_dim = int(noise_dim)
        self.generator_hidden_dims = tuple(int(v) for v in generator_hidden_dims)
        self.discriminator_hidden_dims = tuple(int(v) for v in discriminator_hidden_dims)
        self.dropout = float(dropout)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.learning_rate = float(learning_rate)
        self.random_state = int(random_state)
        self.device = device
        self.verbose = bool(verbose)

        self.columns: list[str] = []
        self.output_dim: int | None = None
        self.scaler: StandardScaler | None = None
        self.generator = None
        self.discriminator = None
        self.loss_history: dict[str, list[float]] = {"generator": [], "discriminator": []}

    def fit(self, full_df: pd.DataFrame):
        if torch is None or DataLoader is None or TensorDataset is None:
            raise ImportError("PyTorch is required for CustomTabularGANSynthesizer. Install torch first.")

        self.columns = list(full_df.columns)
        numeric_df = full_df.apply(pd.to_numeric, errors="coerce").fillna(0)
        train_array = numeric_df.to_numpy(dtype=np.float32)

        self.output_dim = train_array.shape[1]
        self.scaler = StandardScaler()
        train_scaled = self.scaler.fit_transform(train_array).astype(np.float32)

        selected_device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        torch_device = torch.device(selected_device)

        self.generator = TabularGenerator(
            noise_dim=self.noise_dim,
            output_dim=self.output_dim,
            hidden_dims=self.generator_hidden_dims,
        ).to(torch_device)

        self.discriminator = TabularDiscriminator(
            input_dim=self.output_dim,
            hidden_dims=self.discriminator_hidden_dims,
            dropout=self.dropout,
        ).to(torch_device)

        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)

        train_tensor = torch.as_tensor(train_scaled, dtype=torch.float32)
        train_dataset = TensorDataset(train_tensor)

        loader_generator = torch.Generator()
        loader_generator.manual_seed(self.random_state)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=loader_generator,
        )

        criterion = nn.BCELoss()
        generator_optimizer = torch.optim.Adam(
            self.generator.parameters(),
            lr=self.learning_rate,
        )
        discriminator_optimizer = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.learning_rate,
        )

        self.generator.train()
        self.discriminator.train()
        self.loss_history = {"generator": [], "discriminator": []}

        for epoch in range(1, self.epochs + 1):
            generator_losses = []
            discriminator_losses = []

            for (real_batch,) in train_loader:
                real_batch = real_batch.to(torch_device)
                current_batch_size = real_batch.size(0)

                real_labels = torch.ones(current_batch_size, device=torch_device)
                fake_labels = torch.zeros(current_batch_size, device=torch_device)

                discriminator_optimizer.zero_grad()

                real_pred = self.discriminator(real_batch)
                real_loss = criterion(real_pred, real_labels)

                z = torch.randn(current_batch_size, self.noise_dim, device=torch_device)
                fake_batch = self.generator(z)
                fake_pred = self.discriminator(fake_batch.detach())
                fake_loss = criterion(fake_pred, fake_labels)

                discriminator_loss = real_loss + fake_loss
                discriminator_loss.backward()
                discriminator_optimizer.step()

                generator_optimizer.zero_grad()

                z = torch.randn(current_batch_size, self.noise_dim, device=torch_device)
                fake_batch = self.generator(z)
                fake_pred = self.discriminator(fake_batch)
                generator_loss = criterion(fake_pred, real_labels)

                generator_loss.backward()
                generator_optimizer.step()

                discriminator_losses.append(float(discriminator_loss.item()))
                generator_losses.append(float(generator_loss.item()))

            epoch_discriminator_loss = float(np.mean(discriminator_losses))
            epoch_generator_loss = float(np.mean(generator_losses))
            self.loss_history["discriminator"].append(epoch_discriminator_loss)
            self.loss_history["generator"].append(epoch_generator_loss)

            if self.verbose and (epoch == 1 or epoch % 50 == 0 or epoch == self.epochs):
                print(
                    f"Epoch {epoch:04d}/{self.epochs}: "
                    f"d_loss={epoch_discriminator_loss:.4f}, "
                    f"g_loss={epoch_generator_loss:.4f}"
                )

        return self

    def sample(self, num_rows: int) -> pd.DataFrame:
        if torch is None:
            raise ImportError("PyTorch is required for CustomTabularGANSynthesizer. Install torch first.")
        if self.generator is None or self.scaler is None or self.output_dim is None:
            raise RuntimeError("Please call fit() before sample().")

        selected_device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        torch_device = torch.device(selected_device)

        self.generator.eval()
        with torch.no_grad():
            z = torch.randn(int(num_rows), self.noise_dim, device=torch_device)
            generated_scaled = self.generator(z).detach().cpu().numpy()

        generated = self.scaler.inverse_transform(generated_scaled)
        return pd.DataFrame(generated, columns=self.columns)


def build_synthesizer(
    method: str,
    metadata: SingleTableMetadata,
    *,
    epochs: int = 500,
    batch_size: int = 500,
    cuda: bool = True,
    **kwargs: Any,
):
    method_key = method.upper()

    if method_key == "CTGAN":
        return CTGANSynthesizer(
            metadata,
            epochs=epochs,
            batch_size=batch_size,
            cuda=cuda,
            **kwargs,
        )

    if method_key == "TVAE":
        return TVAESynthesizer(
            metadata,
            epochs=epochs,
            batch_size=batch_size,
            cuda=cuda,
            **kwargs,
        )

    if method_key in {"CUSTOM_GAN", "CUSTOMGAN", "GAN"}:
        return CustomTabularGANSynthesizer(
            epochs=epochs,
            batch_size=batch_size,
            **kwargs,
        )

    supported = ", ".join(SUPPORTED_SYNTHESIZERS)
    raise ValueError(f"Unsupported synthesizer method: {method}. Supported methods: {supported}.")
