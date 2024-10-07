import os, sys, random

from PIL import Image, ImageOps
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

def prepare_samples(device: str, sample_dir: str = "./samples/", splits: tuple[float, float, float] = (0.75, 0.15, 0.10), seed = 42):
    print("Discovering samples... ", end="", flush=True)

    all_filenames = os.listdir(sample_dir)
    prefixes = {name.split(".", 1)[0] for name in all_filenames}

    dataset_filenames = [
        ([os.path.join(sample_dir, fn) for fn in files], int(files[0].split(".")[2]))
        for files in
        [
            sorted([fn for fn in all_filenames if fn.split(".", 1)[0] == prefix])
            for prefix in prefixes
        ]
    ]

    print(f"loading {len(dataset_filenames)} image pairs... ", end="", flush=True)

    def load_image(filename: str):
        image = Image.open(filename)
        image.load()
        return image

    def images_to_numpy(a: Image.Image, b: Image.Image):
        a, b = np.array(a), np.array(b)
        return np.dstack((a, b)).astype(np.float32) / 255.0

    def load_sample(image_filenames: tuple[str, str], flip: tuple[bool, bool] = (False, False), angle: float = 0):
        images = [load_image(filename) for filename in image_filenames]
        if flip[0]: images = [ImageOps.flip(image) for image in images]
        if flip[1]: images = [ImageOps.mirror(image) for image in images]
        if angle != 0: images = [image.rotate(angle) for image in images]
        return images_to_numpy(*images)

    def random_augment_params():
        return (random.random() < 0.2, random.random() < 0.2), random.choice([0, 90, 180, 270])

    augment_factor = 4
    dataset = [
        (load_sample(images, *random_augment_params()), label / 100)
        for images, label in dataset_filenames
        for _ in range(augment_factor)
    ]
    dataset_images, dataset_labels = zip(*dataset)

    print("done.")

    print(f"With augmentation, there are now {len(dataset)} samples.")

    # We now have a dataset of (image, label) tuples, where each image is a Numpy array of shape (128, 128, 6).

    print("Uploading samples to device...")
    tensor_images = torch.Tensor(np.array(dataset_images)).permute(0, 3, 1, 2).to(device) # (N, 128, 128, 6) -> [N, 6, 128, 128]
    tensor_labels = torch.Tensor(dataset_labels).to(device)
    tensor_dataset = TensorDataset(tensor_images, tensor_labels)

    print("Splitting dataset...")
    split_generator = torch.Generator().manual_seed(seed)
    train_dataset, test_dataset, val_dataset = torch.utils.data.random_split(tensor_dataset, splits, generator=split_generator)
    print(f"Split: {len(train_dataset)} train / {len(test_dataset)} test / {len(val_dataset)} val samples.")
    train_dataloader = DataLoader(train_dataset)
    test_dataloader = DataLoader(test_dataset)
    val_dataloader = DataLoader(val_dataset)

    return train_dataloader, test_dataloader, val_dataloader

class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=6, out_channels=32, kernel_size=5, padding=2, bias=False)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, bias=False)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.LazyLinear(512)
        self.fc2 = nn.LazyLinear(64)
        self.fc3 = nn.LazyLinear(1)
        self.dropout = nn.Dropout(0.35)

    def forward(self, x):
        x = self.max_pool(F.relu(self.bn1(self.conv1(x))))
        x = self.max_pool(F.relu(self.bn2(self.conv2(x))))
        x = self.max_pool(F.relu(self.bn3(self.conv3(x))))
        x = torch.flatten(x, 1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = F.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x))

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X).squeeze(1)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if batch % 256 == 0 or batch == size - 1:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"[{current:>5d}/{size:>5d}]: loss {loss:>7f}")

def test(title, dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    loss, mae = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X).squeeze(1)
            loss += loss_fn(pred, y).item()
            mae += torch.mean(torch.abs(pred - y))
    loss /= num_batches
    mae /= size
    if title is not None:
        print(f"{title:>10}: mae {100 * mae:>3.0f}% / avg loss {loss:>8f}")
    return loss, mae

if __name__ == "__main__":
    train_flag_present = "--train" in sys.argv
    test_flag_present = "--test" in sys.argv
    if train_flag_present == test_flag_present:
        print("Please pass either --train or --test to this script.")
        exit()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}.")

    train_dataloader, test_dataloader, val_dataloader = prepare_samples(device, splits=(0.5, 0.4, 0.1))
    loss_func = nn.MSELoss()

    if train_flag_present:
        # Train the model
        model = CNNModel()
        print("Model:")
        print(model)
        model.to(device)
        epochs = 32
        best_loss = 1e9
        run_name = "".join(random.choices("abcdef09123456789", k=8))
        optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
        for t in range(epochs):
            print("=" * 32)
            print(f"Epoch {t + 1}")
            print("=" * 32)
            train(train_dataloader, model, loss_func, optimizer)
            test("Training", train_dataloader, model, loss_func)
            loss, mae = test("Testing", test_dataloader, model, loss_func)
            if loss < best_loss:
                model_filename = f"best-{100 * mae:.0f}-mae-{run_name}.pt"
                print(f"New best loss! Saving model as '{model_filename}'...")
                torch.save(model, model_filename)
                best_loss = loss
            scheduler.step(loss)
            print(f"Current learning rate is {optimizer.param_groups[0]['lr']}.")
        print("Training finished!")
    else:
        # Load and test the model
        import warnings
        with warnings.catch_warnings():
            # We silence the warning about arbitrary code execution. However, please note that you
            # should still be careful about the model that your are loading. Only load trusted
            # models.
            warnings.simplefilter("ignore")
            model = torch.load("model.pt", weights_only=False)
        loss, mae = test(None, test_dataloader, model, loss_func)
        print(f"Test result: loss = {loss}, MAE = {mae}")
