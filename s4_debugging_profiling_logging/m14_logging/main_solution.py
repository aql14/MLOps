import matplotlib.pyplot as plt
import torch
import typer
import wandb

from data_solution import corrupt_mnist
from model_solution import MyAwesomeModel

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

app = typer.Typer()


@app.command()
def train(lr: float = 1e-3, batch_size: int = 32, epochs: int = 5) -> None:
    """Train a model on MNIST."""
    print("Training day and night")
    print(f"{lr=}, {batch_size=}, {epochs=}")

    # --- wandb init (captures hyperparameters) ---
    run = wandb.init(
        project="corrupt_mnist",
        config={"lr": lr, "batch_size": batch_size, "epochs": epochs},
    )

    model = MyAwesomeModel().to(DEVICE)
    train_set, _ = corrupt_mnist()
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    statistics = {"train_loss": [], "train_accuracy": []}

    global_step = 0
    for epoch in range(epochs):
        model.train()
        for i, (img, target) in enumerate(train_dataloader):
            img, target = img.to(DEVICE), target.to(DEVICE)

            optimizer.zero_grad()
            y_pred = model(img)
            loss = loss_fn(y_pred, target)
            loss.backward()
            optimizer.step()

            accuracy = (y_pred.argmax(dim=1) == target).float().mean().item()

            statistics["train_loss"].append(loss.item())
            statistics["train_accuracy"].append(accuracy)

            # --- wandb scalar logging each step ---
            wandb.log(
                {"train_loss": loss.item(), "train_accuracy": accuracy, "epoch": epoch},
                step=global_step,
            )
            global_step += 1

            if i % 100 == 0:
                print(f"Epoch {epoch}, iter {i}, loss: {loss.item()}")

    print("Training complete")

    # Save model locally (you already do this)
    model_path = "model.pth"
    torch.save(model.state_dict(), model_path)

    # --- log model as a wandb Artifact ---
    final_train_accuracy = float(statistics["train_accuracy"][-1]) if statistics["train_accuracy"] else 0.0

    artifact = wandb.Artifact(
        name="corrupt_mnist_model",
        type="model",
        description="MyAwesomeModel trained on corrupt MNIST",
        metadata={
            "lr": lr,
            "batch_size": batch_size,
            "epochs": epochs,
            "final_train_accuracy": final_train_accuracy,
            "device": str(DEVICE),
        },
    )
    artifact.add_file(model_path)
    run.log_artifact(artifact)

    # Keep your plots
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(statistics["train_loss"])
    axs[0].set_title("Train loss")
    axs[1].plot(statistics["train_accuracy"])
    axs[1].set_title("Train accuracy")
    fig.savefig("training_statistics.png")
    plt.close(fig)

    wandb.finish()


@app.command()
def evaluate(model_checkpoint: str) -> None:
    """Evaluate a trained model."""
    print("Evaluating like my life depended on it")
    print(model_checkpoint)

    model = MyAwesomeModel().to(DEVICE)
    model.load_state_dict(torch.load(model_checkpoint, map_location=DEVICE))

    _, test_set = corrupt_mnist()
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=32)

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for img, target in test_dataloader:
            img, target = img.to(DEVICE), target.to(DEVICE)
            y_pred = model(img)
            correct += (y_pred.argmax(dim=1) == target).float().sum().item()
            total += target.size(0)

    print(f"Test accuracy: {correct / total}")


if __name__ == "__main__":
    app()
