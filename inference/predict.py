import torch

from models.unet import UNet



def load_model(path):

    model = UNet()

    model.load_state_dict(
        torch.load(path)
    )

    model.eval()

    return model




def predict(
        model,
        image
):

    image = torch.tensor(
        image,
        dtype=torch.float32
    )


    image = image.unsqueeze(0)


    with torch.no_grad():

        pred = model(image)


    mask = (
        pred.squeeze()
        .numpy()
        >0.5
    )


    confidence = (
        pred.squeeze()
        .numpy()
    )


    return mask, confidence