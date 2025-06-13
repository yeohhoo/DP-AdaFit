from ..utils.dataset import (get_data_loader, get_data_loader_augmented,
                     getImagenetTransform, build_transform)


def prepare_dataloaders(args):
    """Data loader capable of supporting data augmentation
    """
    if args.transform:
        transforms = []
        if "deit" in args.type_of_augmentation:
            for _ in range(args.transform):
                transforms.append(build_transform(True, args))
        else:
            for _ in range(args.transform):
                transforms.append(
                    getImagenetTransform(args.type_of_augmentation,
                                         img_size=args.img_size, #
                                         crop_size=args.crop_size, #
                                         normalization=True,
                                         as_list=False,
                                         differentiable=False,
                                         params=None))
        train_loader = get_data_loader_augmented(
            args,
            split='train',
            transforms=transforms,
            shuffle=True,
        )
    else:
        train_loader = get_data_loader(
            args,
            split='train',
            transform='center',#args.train_transform,
            shuffle=True
        )

    test_loader = get_data_loader(
        args,
        split='valid',
        transform='center',
        shuffle=False
    )
    return train_loader, test_loader
