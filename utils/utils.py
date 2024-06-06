from pathlib import Path
import matplotlib.pyplot as plt
from IPython.display import clear_output


def get_project_root() -> Path:
    return Path(__file__).parent.parent


def update_plot(epoch, train_losses):
    clear_output(wait=False)
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Epoch {epoch + 1}')
    plt.legend()
    plt.grid(True)
    plt.show()
