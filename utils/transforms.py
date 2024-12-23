import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transforms():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
        A.CoarseDropout(max_holes=1, max_height=16, max_width=16, min_holes=1, 
                       min_height=16, min_width=16, fill_value=[0.4914, 0.4822, 0.4465], p=0.5),
        A.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),
        ToTensorV2()
    ])

def get_test_transforms():
    return A.Compose([
        A.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),
        ToTensorV2()
    ]) 