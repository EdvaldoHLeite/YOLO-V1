import matplotlib.pyplot as plt
import torch
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

    visualize_dataloader_batch(train_dataloader)


def visualize_dataloader_batch(dataloader, num_images=16):
    images, targets = next(iter(dataloader))

    classes = [
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "motorbike",
        "person",
        "pottedplant",
        "sheep",
        "sofa",
        "train",
        "tvmonitor",
    ]
    cell_size = 448.0 / 7.0

    for idx in range(min(num_images, images.shape[0])):
        fig, ax = plt.subplots(figsize=(6, 6))

        img = images[idx].permute(1, 2, 0).numpy()
        target = targets[idx]

        ax.imshow(img)
        ax.set_title(f"Sample {idx}")

        # Draw bounding boxes
        for i in range(7):
            for j in range(7):
                conf = target[j, i, 4].item()
                if conf > 0.5:
                    x_cell, y_cell, w_cell, h_cell = target[j, i, 0:4].tolist()

                    box_x = (x_cell + i) * cell_size
                    box_y = (y_cell + j) * cell_size
                    box_w = w_cell * 448.0
                    box_h = h_cell * 448.0

                    xmin = box_x - (box_w / 2.0)
                    ymin = box_y - (box_h / 2.0)

                    class_idx = torch.argmax(target[j, i, 10:]).item()
                    class_name = classes[class_idx]

                    rect = plt.Rectangle(
                        (xmin, ymin),
                        box_w,
                        box_h,
                        linewidth=2,
                        edgecolor="red",
                        facecolor="none",
                    )
                    ax.add_patch(rect)
                    ax.text(
                        xmin,
                        ymin - 2,
                        class_name,
                        color="white",
                        fontsize=8,
                        backgroundcolor="red",
                    )

        plt.savefig(f"test/samples/yolo_sample_{idx}.png")
        plt.close(fig)  # Close the figure to free up memory
    """ plt.tight_layout()
    plt.savefig("test/preview/yolo_batch_preview.png")
    print("Saved preview visualization to test/yolo_batch_preview.png!") """
