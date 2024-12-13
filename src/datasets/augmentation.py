from torchvision import transforms as tfs


def get_94percent_augmentation(h=32, w=32, crop_size=4):
    crop = tfs.RandomCrop((h, w), padding=crop_size, padding_mode="reflect")
    flip = tfs.RandomHorizontalFlip()
    erase = tfs.RandomErasing(p=1., scale=(1 / 16, 1 / 16), ratio=(1., 1.))
    return tfs.Compose([crop, flip, erase])


AUGMENTATIONS = {
    "94perc": get_94percent_augmentation(),
    "None": lambda x: x,
}


names = list(AUGMENTATIONS.keys())
get = AUGMENTATIONS.get
