from torch.utils.data import Dataset
import cv2

class SegmentationDataset(Dataset):

    def __init__(self, image_paths, mask_paths, transforms=None):
        assert len(image_paths) == len(mask_paths)
        self.image_paths = image_paths[:]
        self.mask_paths =  mask_paths[:]
        self.transforms = transforms


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask_path = self.mask_paths[idx]
        mask = cv2.imread(mask_path, 0)

        if self.transforms:
            image = self.transforms(image)
            mask = self.transforms(mask)

        return (image, mask)


if __name__ == '__main__':
    import os
    from imutils.paths import list_images

    data_root = os.path.expanduser('~/ml_datasets/')
    path_to_dataset = os.path.join(data_root, 'TGS_salt')

    IMAGE_DATASET_PATH = os.path.join(path_to_dataset, "train/images")
    image_paths = sorted(list(list_images(IMAGE_DATASET_PATH)))

    MASK_DATASET_PATH = os.path.join(path_to_dataset, "train/masks")
    mask_paths = sorted(list(list_images(MASK_DATASET_PATH)))


    dataset = SegmentationDataset(image_paths, mask_paths)

    idx = 17
    image, mask = dataset[idx]
    print(image.shape)
    print(mask.shape)







