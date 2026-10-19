import matplotlib.pyplot as plt


def plot_training_losses(history):
    plt.plot(history["generator"], label="Generator")
    plt.plot(history["discriminator"], label="Discriminator")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.show()
