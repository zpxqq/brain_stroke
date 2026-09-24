import torch
import numpy as np
from scipy import ndimage

from models.unet import UNet


def load_model(path):
    model = UNet()

    # Streamlit Cloud обычно работает на CPU,
    # поэтому явно загружаем веса на CPU
    state_dict = torch.load(
        path,
        map_location=torch.device("cpu")
    )

    model.load_state_dict(state_dict)

    model.eval()

    return model


def predict(model, image):
    image = torch.tensor(
        image,
        dtype=torch.float32
    )

    # Добавляем batch dimension
    image = image.unsqueeze(0)

    with torch.no_grad():
        pred = model(image)

        # Если модель возвращает logits
        pred = torch.sigmoid(pred)

    confidence = pred.squeeze().cpu().numpy()

    mask = confidence > 0.5

    return mask, confidence


def count_lesions(mask, min_voxels=10):
    """
    Подсчитывает количество отдельных очагов
    на бинарной маске.

    mask:
        numpy-массив 2D или 3D

    min_voxels:
        минимальный размер очага.
        Очень маленькие компоненты считаем шумом.
    """

    mask = np.asarray(mask)
    mask = np.squeeze(mask)

    binary_mask = mask > 0.5

    # Находим отдельные связные области
    labeled_mask, num_components = ndimage.label(binary_mask)

    lesion_count = 0

    for component_id in range(1, num_components + 1):

        component_size = np.sum(
            labeled_mask == component_id
        )

        if component_size >= min_voxels:
            lesion_count += 1

    return lesion_count
