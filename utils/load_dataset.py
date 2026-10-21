# import xml.etree.ElementTree as ET

import torch
import torchvision.transforms as T
from torchvision.datasets import VOCDetection

""" VOC_dataset = VOCDetection(
    root="data",
    year="2012",
    image_set="trainval",
    download=True,
    transform=T.Compose(
        [
            T.ToTensor(),
            T.Resize((448, 448)),  # image size is (448, 448)
        ]
    ),
) """


class YOLOVOCDataset(torch.utils.data.Dataset):
    def __init__(
        self, root, year, image_set, split_size=7, num_boxes=2, num_classes=20
    ):
        # 1. Initialize the official torchvision VOC dataset
        self.voc = VOCDetection(
            root=root,
            year=year,
            image_set=image_set,
            download=False,  # download is set to False because the data was already saved and to not extract again
        )

        self.split_size = split_size
        self.num_boxes = num_boxes
        self.num_classes = num_classes

        # PASCAL VOC class mapping
        self.classes = [
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
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        # Image transformation to 448x448
        self.transform = T.Compose([T.Resize((448, 448)), T.ToTensor()])

    def __len__(self):
        return len(self.voc)

    def __getitem__(self, idx):
        # VOCDetection returns a tuple: (PIL_Image, target_dict)
        image, target_dict = self.voc[idx]

        # Original image width and height (needed to scale bounding boxes)
        orig_w = float(target_dict["annotation"]["size"]["width"])
        orig_h = float(target_dict["annotation"]["size"]["height"])

        # 1. Resize image to 448x448 and convert to tensor
        image = self.transform(image)

        # 2. Extract and scale bounding boxes to the 448x448 space
        boxes = []
        for obj in target_dict["annotation"]["object"]:
            # Skip difficult objects if desired
            if int(obj.get("difficult", 0)) == 1:
                continue

            class_name = obj["name"]
            if class_name not in self.class_to_idx:
                continue
            class_idx = self.class_to_idx[class_name]

            bndbox = obj["bndbox"]
            xmin = float(bndbox["xmin"]) / orig_w * 448
            ymin = float(bndbox["ymin"]) / orig_h * 448
            xmax = float(bndbox["xmax"]) / orig_w * 448
            ymax = float(bndbox["ymax"]) / orig_h * 448

            boxes.append([xmin, ymin, xmax, ymax, class_idx])

        # 3. Convert VOC boxes into the YOLOv1 target tensor (7 x 7 x 30)
        target_tensor = self.encode_boxes_to_yolo(boxes)

        return image, target_tensor

    def encode_boxes_to_yolo(self, boxes):
        # Initialize an empty (7, 7, 30) tensor for this image
        # 30 = (2 boxes * 5 coordinates) + 20 classes
        target_matrix = torch.zeros(
            (self.split_size, self.split_size, self.num_boxes * 5 + self.num_classes)
        )

        cell_size = 448.0 / self.split_size  # 64 pixels per grid cell

        for box in boxes:
            xmin, ymin, xmax, ymax, class_idx = box

            # Calculate center, width, and height of the bounding box
            box_x = (xmin + xmax) / 2.0
            box_y = (ymin + ymax) / 2.0
            box_w = xmax - xmin
            box_h = ymax - ymin

            # Find which 7x7 grid cell the center falls into
            i = int(box_x // cell_size)
            j = int(box_y // cell_size)

            # Clamp to prevent out-of-bounds errors
            i = min(i, self.split_size - 1)
            j = min(j, self.split_size - 1)

            # Check if this cell already has an object assigned to it (YOLOv1 simplification)
            if target_matrix[j, i, 4] == 0:
                # Normalize center relative to the grid cell (0 to 1)
                x_cell = (box_x / cell_size) - i
                y_cell = (box_y / cell_size) - j
                # Normalize width and height relative to the whole image (0 to 1)
                w_cell = box_w / 448.0
                h_cell = box_h / 448.0

                # Assign to the first bounding box of this cell
                target_matrix[j, i, 0:4] = torch.tensor(
                    [x_cell, y_cell, w_cell, h_cell]
                )
                target_matrix[j, i, 4] = 1.0  # Confidence score = 1

                # One-hot encode the class probabilities
                target_matrix[j, i, 10 + int(class_idx)] = 1.0

        return target_matrix
