from torch.utils.data import DataLoader

from utils import YOLOVOCDataset


def test_dataset():
    # 1. Instantiate your custom datasets
    yolo_train_dataset = YOLOVOCDataset(root="data", year="2012", image_set="trainval")
    yolo_test_dataset = YOLOVOCDataset(root="data", year="2012", image_set="val")

    # 2. Pass them directly into your DataLoaders
    train_dataloader = DataLoader(
        yolo_train_dataset,
        batch_size=16,
        num_workers=4,
        persistent_workers=True,
        drop_last=True,
        shuffle=True,
    )

    test_dataloader = DataLoader(
        yolo_test_dataset,
        batch_size=16,
        num_workers=4,
        persistent_workers=True,
        drop_last=True,
    )

    # 3. Test the shapes
    initial_batch = next(iter(train_dataloader))
    images, targets = initial_batch[0], initial_batch[1]

    print(f"the shape of batched images: {images.numpy().shape}")
    # Output: (16, 3, 448, 448)

    print(f"the shape of batched targets: {targets.numpy().shape}")
    # Output: (16, 7, 7, 30)
