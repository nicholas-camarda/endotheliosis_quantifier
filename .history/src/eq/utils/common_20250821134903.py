"""Common utilities shared across eq modules."""

import os
import pickle
from pathlib import Path
from typing import Any, Union

import matplotlib.pyplot as plt
import psutil


def load_pickled_data(file_path: Union[str, Path]) -> Any:
    """Load and return a Python object from a pickle file."""
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def print_memory_usage() -> None:
    """Print the current process RSS memory usage in MB."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print(f"Memory usage: {mem_info.rss / (1024 * 1024):.2f} MB")


def plot_history(history, output_dir: Union[str, Path], file_name: str) -> None:
    """Plot training history (loss and accuracy curves) and save to disk."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Loss curves
    loss = history.history.get('loss', [])
    val_loss = history.history.get('val_loss', [])
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'y', label='Training loss')
    if val_loss:
        plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(output_dir / f"{file_name}_loss.png")
    plt.clf()

    # Accuracy curves
    acc = history.history.get('accuracy', [])
    val_acc = history.history.get('val_accuracy', [])
    if acc:
        plt.plot(epochs, acc, 'y', label='Training acc')
        if val_acc:
            plt.plot(epochs, val_acc, 'r', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig(output_dir / f"{file_name}_accuracy.png")
        plt.clf()
        plt.close()
