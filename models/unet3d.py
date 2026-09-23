import torch

from monai.networks.nets import UNet


def create_unet3d():

    model = UNet(

        # 3D медицинские изображения
        spatial_dims=3,


        # DWI + ADC + FLAIR
        in_channels=3,


        # одна маска очага
        out_channels=1,


        # количество каналов внутри сети
        channels=(
            8,
            16,
            32,
            64,
            128
        ),


        # уменьшение размера
        strides=(
            2,
            2,
            2,
            2
        ),


        # дополнительные блоки
        num_res_units=2

    )


    return model



if __name__ == "__main__":


    model = create_unet3d()


    x = torch.randn(
        1,
        3,
        96,
        96,
        96
    )


    y = model(x)


    print(
        "Input:",
        x.shape
    )


    print(
        "Output:",
        y.shape
    )