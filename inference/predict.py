import torch

from models.unet3d import create_unet3d



def load_model(path):

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    model = create_unet3d()

    checkpoint = torch.load(
        path,
        map_location=device
    )

    model.load_state_dict(checkpoint)

    model.to(device)

    model.eval()

    return model, device



def predict(
        model,
        image,
        device
):

    image = torch.tensor(
        image,
        dtype=torch.float32
    )

    # добавляем batch dimension
    image = image.unsqueeze(0)

    # отправляем на GPU
    image = image.to(device)


    with torch.no_grad():

        pred = model(image)


    probability = (
        torch.sigmoid(pred)
        .squeeze()
        .cpu()
        .numpy()
    )


    mask = (
        probability > 0.5
    )


    return mask, probability