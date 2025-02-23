import torch
from functools import cache
import fire
import multiprocessing as mp
import librosa
from torch import nn
from .model import VQVAE
from rich.progress import Progress, TextColumn, BarColumn, TimeRemainingColumn
from rich.console import Console


@cache
def _load(path):
    y, sr = librosa.load(path, sr=None, mono=True)
    y = torch.from_numpy(y).float()
    return y


def _unglob(paths, batch_size, dim):
    chunk_size = batch_size * dim
    total = 0
    waves = []
    with mp.Pool(mp.cpu_count() - 1) as pool:
        for y in pool.imap(_load, paths):
            while y.numel() > chunk_size:
                yield y[:chunk_size]
                y = y[chunk_size:]
            if y.numel() + total > chunk_size:
                padding = chunk_size - total % chunk_size
                waves[-1] = torch.cat([waves[-1], torch.zeros(padding)])
                batch = torch.cat(waves)
                yield batch.reshape(-1, dim)
                waves = [y]
                total = y.numel()
            else:
                waves.append(y)
                total += y.numel()


def main(corpus_path, batch_size: int = 1024, steps: int = 10000) -> None:
    import polars as pl

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vqvae = VQVAE().to(device)
    optim = torch.optim.Adam(vqvae.parameters(), lr=1e-3)
    paths = (
        pl.scan_csv(f"{corpus_path}/validated.tsv", separator="\t")
        .select(pl.lit(f"{corpus_path}/clips/") + pl.col("path"))
        .sort(pl.all().hash())
        .collect()
        .to_series()
    )

    # Initialize EMA
    ema_loss = 0
    beta = 0.99
    step = 0

    console = Console()
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        TextColumn("Loss: {task.fields[loss]:.3g}"),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Training...", total=steps, loss=0.0)

        for x in _unglob(paths, batch_size, vqvae.dim):
            x = x.to(device)
            z = vqvae.encode(x)
            y = vqvae.decode(z)
            optim.zero_grad()
            loss = nn.functional.mse_loss(y, x)
            loss.backward()
            optim.step()

            # Update EMA with bias correction
            loss_val = loss.item()
            ema_loss = beta * ema_loss + (1 - beta) * loss_val
            bias_correction = 1 - beta ** (step + 1)
            corrected_loss = ema_loss / bias_correction

            # Update progress bar
            progress.update(task, advance=1, loss=corrected_loss)

            step += 1
            if step >= steps:
                break

        # Save final model state
        torch.save(vqvae.state_dict(), "vqvae.pt")


if __name__ == "__main__":
    fire.Fire(main)
