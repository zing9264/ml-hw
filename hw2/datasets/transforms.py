from torchvision import transforms

def build_transform(cfg):
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0),
        transforms.Resize(cfg.DATA.RESIZE),
        transforms.ToTensor(),
        transforms.Normalize(cfg.DATA.PIXEL_MEAN,cfg.DATA.PIXEL_STD)
        ])
    return transform


'''
    transform = transforms.Compose([transforms.Resize(cfg.DATA.RESIZE),
                                     transforms.ToTensor(),
                                     transforms.Normalize(cfg.DATA.PIXEL_MEAN,
                                                          cfg.DATA.PIXEL_STD)])

        transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0),
        transforms.Resize(cfg.DATA.RESIZE),
        transforms.ToTensor(),
        transforms.Normalize(cfg.DATA.PIXEL_MEAN,cfg.DATA.PIXEL_STD)
        ])                                                      
'''
