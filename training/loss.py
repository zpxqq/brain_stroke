import torch
import torch.nn as nn



bce = nn.BCEWithLogitsLoss()



def dice_loss(pred, target):

    pred = torch.sigmoid(pred)


    smooth = 1


    pred = pred.reshape(-1)
    target = target.reshape(-1)


    intersection = (
        pred * target
    ).sum()


    dice = (
        2 * intersection + smooth
    ) / (
        pred.sum()
        +
        target.sum()
        +
        smooth
    )


    return 1 - dice




def combined_loss(pred, target):

    return (
        bce(pred, target)
        +
        dice_loss(pred, target)
    )