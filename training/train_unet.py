import torch
from torch.utils.data import DataLoader, random_split

from dataset import ISLESDataset
from loss import dice_loss
from models.unet import UNet
from training.loss import combined_loss
device = "cuda" if torch.cuda.is_available() else "cpu"
import os

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

WEIGHTS_PATH = os.path.join(
    BASE_DIR,
    "weights",
    "unet_best.pth"
)

# =====================
# Dataset
# =====================

dataset = ISLESDataset(
    "C:/AI_brai/ISLES-2022"
)


train_size = int(len(dataset)*0.8)

val_size = len(dataset)-train_size


train_dataset, val_dataset = random_split(
    dataset,
    [train_size,val_size]
)



train_loader = DataLoader(
    train_dataset,
    batch_size=2,
    shuffle=True
)


val_loader = DataLoader(
    val_dataset,
    batch_size=2,
    shuffle=False
)



# =====================
# Model
# =====================

model = UNet(
    in_channels=3,
    out_channels=1
)


model.to(device)



optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.0001
)



# =====================
# Training
# =====================

epochs = 20


best_loss = 999



for epoch in range(epochs):

    model.train()

    train_loss = 0


    for images, masks in train_loader:


        images = images.to(device)

        masks = masks.to(device)


        prediction = model(images)

        loss = combined_loss(
            prediction,
            masks
        )


        optimizer.zero_grad()

        loss.backward()

        optimizer.step()


        train_loss += loss.item()



    train_loss /= len(train_loader)



    # =====================
    # Validation
    # =====================

    model.eval()

    val_loss = 0


    with torch.no_grad():

        for images,masks in val_loader:

            images = images.to(device)

            masks = masks.to(device)


            prediction = model(images)


            loss = dice_loss(
                prediction,
                masks
            )


            val_loss += loss.item()



    val_loss /= len(val_loader)



    print(
        f"""
Epoch {epoch+1}/{epochs}

Train loss:
{train_loss}

Val loss:
{val_loss}
"""
    )


    if val_loss < best_loss:

        best_loss = val_loss

        torch.save(
            model.state_dict(),
            WEIGHTS_PATH
        )

        print("Модель сохранена")