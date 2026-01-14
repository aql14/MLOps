import torch
from torch.utils.data import TensorDataset


def test_training_creates_artifacts(tmp_path, monkeypatch):
    # Run training in an isolated temporary folder
    monkeypatch.chdir(tmp_path)

    # Create expected output folders (your script assumes they exist)
    (tmp_path / "models").mkdir(parents=True, exist_ok=True)
    (tmp_path / "reports" / "figures").mkdir(parents=True, exist_ok=True)

    # Import training module
    import s5_ci.train as train_module

    # ---- Mock corrupt_mnist so we don't depend on real data files ----
    x = torch.randn(128, 1, 28, 28)              # 128 fake images
    y = torch.randint(0, 10, (128,))             # 128 fake labels
    fake_train = TensorDataset(x, y)
    fake_test = TensorDataset(x[:32], y[:32])

    monkeypatch.setattr(train_module, "corrupt_mnist", lambda: (fake_train, fake_test))

    # Run a very small training job
    train_module.train(lr=1e-3, batch_size=64, epochs=1)

    # Check artifacts were created
    assert (tmp_path / "models" / "model.pth").exists()
    assert (tmp_path / "reports" / "figures" / "training_statistics.png").exists()
