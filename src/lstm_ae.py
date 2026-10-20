"""LSTM autoencoder detector for GridGuard.

The model sees one day (48 half-hours) of a household's readings, scaled by that
household's usual level, and learns to reconstruct normal days. Days it cannot
reconstruct well are flagged. It has the same interface as the baselines:
score(df) returns a table where a higher value means "more anomalous".
"""

import numpy as np
import pandas as pd
import torch
from torch import nn

WINDOW = 48        # one day of half-hourly readings
N_FEATURES = 4     # scaled reading, sin(time of day), cos(time of day), weekend flag


def build_features(df, level):
    """Array of shape (time, meter, 4): scaled reading plus time-of-day inputs."""
    rel = (df / level.reindex(df.columns)).clip(0, 8)
    value = np.log1p(rel.to_numpy(dtype="float32"))
    slot = (df.index.hour * 2 + df.index.minute // 30).to_numpy()
    angle = 2 * np.pi * slot / 48
    weekend = np.asarray(df.index.dayofweek >= 5, dtype="float32")
    cond = np.stack([np.sin(angle), np.cos(angle), weekend], axis=1).astype("float32")
    X = np.empty((len(df), df.shape[1], N_FEATURES), dtype="float32")
    X[:, :, 0] = value
    X[:, :, 1:] = cond[:, None, :]
    return X


def window_view(X, length=WINDOW):
    """View of all windows, shape (n_windows, meters, length, features).

    Window k covers rows k .. k+length-1, so it ends at row k+length-1.
    """
    view = np.lib.stride_tricks.sliding_window_view(X, length, axis=0)
    return view.transpose(0, 1, 3, 2)


def training_windows(X, stride, length=WINDOW):
    """Every `stride`-th window of every meter, as one tensor (n, length, features)."""
    view = window_view(X, length)
    picked = view[np.arange(0, view.shape[0], stride)]
    flat = np.ascontiguousarray(picked.reshape(-1, length, X.shape[2]))
    return torch.from_numpy(flat)


class LSTMAutoencoder(nn.Module):
    """Encoder squeezes a day into a small code; decoder rebuilds the readings."""

    def __init__(self, hidden=32, latent=16, n_features=N_FEATURES):
        super().__init__()
        self.encoder = nn.LSTM(n_features, hidden, batch_first=True)
        self.to_latent = nn.Linear(hidden, latent)
        # the decoder also gets time-of-day and weekend inputs (features 1 to 3)
        self.decoder = nn.LSTM(latent + n_features - 1, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):                      # x: (batch, length, features)
        _, (h, _) = self.encoder(x)
        z = torch.relu(self.to_latent(h[-1]))  # (batch, latent)
        z_seq = z.unsqueeze(1).expand(-1, x.size(1), -1)
        decoded, _ = self.decoder(torch.cat([z_seq, x[:, :, 1:]], dim=2))
        return self.head(decoded).squeeze(-1)  # (batch, length)


class LSTMAutoencoderDetector:
    name = "LSTM autoencoder"

    def __init__(self, level, hidden=32, latent=16, seed=42, n_val_meters=20):
        self.level = level                 # per-meter mean reading (from clean history)
        self.n_val_meters = n_val_meters   # last meters of the training set are held out
        torch.manual_seed(seed)
        self.model = LSTMAutoencoder(hidden=hidden, latent=latent)
        self.history = {"train": [], "val": []}

    def _loss(self, batch):
        recon = self.model(batch)
        return ((recon - batch[:, :, 0]) ** 2).mean()

    def fit(self, df, epochs=8, batch_size=256, lr=1e-3, stride=24, verbose=True):
        """Train on clean data (time x meter). The last meters are held out for validation."""
        n_val = self.n_val_meters
        train = training_windows(build_features(df.iloc[:, :-n_val], self.level), stride)
        val = training_windows(build_features(df.iloc[:, -n_val:], self.level), stride)
        if verbose:
            print(f"Training windows: {len(train):,} | validation windows: {len(val):,}")

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        for epoch in range(1, epochs + 1):
            self.model.train()
            order = torch.randperm(len(train))
            total = 0.0
            for i in range(0, len(train), batch_size):
                batch = train[order[i:i + batch_size]]
                loss = self._loss(batch)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                total += loss.item() * len(batch)
            train_loss = total / len(train)

            self.model.eval()
            val_total = 0.0
            with torch.no_grad():
                for j in range(0, len(val), 2048):
                    part = val[j:j + 2048]
                    val_total += self._loss(part).item() * len(part)
            val_loss = val_total / len(val)

            self.history["train"].append(train_loss)
            self.history["val"].append(val_loss)
            if verbose:
                print(f"Epoch {epoch}/{epochs}: train loss {train_loss:.4f} | "
                      f"validation loss {val_loss:.4f}")
        return self

    def score(self, df, step=1, chunk=64):
        """Reconstruction error of the day ending at each reading (time x meter).

        step > 1 scores only every step-th reading (faster, for calibration);
        the other rows are left empty (NaN).
        """
        self.model.eval()
        X = build_features(df, self.level)
        n_rows, n_meters, n_feat = X.shape
        view = window_view(X)
        out = np.full((n_rows, n_meters), np.nan, dtype="float32")
        ends = np.arange(0, view.shape[0], step)
        with torch.no_grad():
            for a in range(0, len(ends), chunk):
                idx = ends[a:a + chunk]
                batch = np.ascontiguousarray(view[idx]).reshape(-1, WINDOW, n_feat)
                tensor = torch.from_numpy(batch)
                recon = self.model(tensor)
                err = ((recon - tensor[:, :, 0]) ** 2).mean(dim=1).numpy()
                out[idx + WINDOW - 1] = err.reshape(len(idx), n_meters)
        scores = pd.DataFrame(out, index=df.index, columns=df.columns)
        return scores.fillna(0.0) if step == 1 else scores

    def save(self, path):
        torch.save(self.model.state_dict(), path)