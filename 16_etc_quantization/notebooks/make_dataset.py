import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2
from PIL import Image


ROOT_DIR = os.path.dirname(os.getcwd())
DATA_DIR = os.path.join(ROOT_DIR, "data")
MODEL_DIR = os.path.join(ROOT_DIR, "models")

RAW_DIR = os.path.join(DATA_DIR, "raw")
IMAGE_DIR = os.path.join(RAW_DIR, "images", "images")
ANNOT_DIR = os.path.join(RAW_DIR, "annotations", "annotations")

BBOX_DIR = os.path.join(ANNOT_DIR, "xmls")
TRAINVAL_LABEL_PATH = os.path.join(ANNOT_DIR, "trainval.txt")
TEST_LABEL_PATH = os.path.join(ANNOT_DIR, "test.txt")


def get_label_dict(txt_file):

    f = open(txt_file, "r")
    lines = f.readlines()

    # \n 제거
    info_list = [line.strip() for line in lines]

    label_dict = {
        item.split(" ")[0]: {
            "class_id": item.split(" ")[1],
            "species": item.split(" ")[2],
            "breed_id": item.split(" ")[3]
        }
        for item in info_list
    }

    return label_dict


def get_bbox(img_path):
    import xml.etree.ElementTree as ET

    with open(img_path) as f:
        tree = ET.parse(f)
        root = tree.getroot()
    
    for obj in root.findall("object"):      
        bbox = obj.find("bndbox")
        xmin = int(bbox.find("xmin").text)
        ymin = int(bbox.find("ymin").text)
        xmax = int(bbox.find("xmax").text)
        ymax = int(bbox.find("ymax").text)
            
    return [xmin, ymin, xmax, ymax]


def get_train_dataset():

    from glob import glob
    import xml.etree.ElementTree as ET
    from sklearn.model_selection import train_test_split
    from torch.utils.data import Subset, DataLoader


    trainval_label_dict = get_label_dict(TRAINVAL_LABEL_PATH)
    test_label_dict = get_label_dict(TEST_LABEL_PATH)

    trainval_label_set = set(trainval_label_dict.keys())
    test_label_set = set(test_label_dict.keys())

    all_image_set = {file_name.split(".")[0].strip() for file_name in os.listdir(IMAGE_DIR)}
    all_bbox_set = {file_name.split(".")[0].strip() for file_name in os.listdir(BBOX_DIR)}

    trainval_image_set = all_image_set & all_bbox_set & trainval_label_set
    test_image_set = all_image_set & all_bbox_set & test_label_set

    trainval_bbox_set = trainval_image_set & all_bbox_set & trainval_label_set
    test_bbox_set = test_image_set & (all_bbox_set - trainval_bbox_set) & test_label_set


    multi_object_name_set = set()

    bbox_path_list = sorted(glob(os.path.join(BBOX_DIR, "*.xml")))

    for path in bbox_path_list:
        with open(path) as f:
            tree = ET.parse(f)
            root = tree.getroot()
        
        objects = root.findall("object")
        num_objects = len(objects)
        
        if num_objects > 1:
            multi_object_name_set.add(path.split("/")[-1].split(".")[0])


    trainval_name_list = sorted(list(trainval_image_set & trainval_label_set & trainval_bbox_set))
    del_name_set = trainval_label_set - trainval_image_set - trainval_bbox_set

    trainval_name_list = [
        name for name in trainval_name_list
        if name not in (del_name_set and multi_object_name_set)
    ]

    TRAINVAL_IMAGE_LIST = [f"{os.path.join(IMAGE_DIR, name)}.jpg" for name in trainval_name_list]
    TRAINVAL_BBOX_LIST = [f"{os.path.join(BBOX_DIR, name)}.xml" for name in trainval_name_list]
    TRAINVAL_LABEL_LIST = [int(trainval_label_dict[name]["species"]) for name in trainval_name_list]


    class CatDogDataset(Dataset):
        def __init__(self):
            self.images = TRAINVAL_IMAGE_LIST
            self.bboxes = TRAINVAL_BBOX_LIST
            self.labels = TRAINVAL_LABEL_LIST
            self.transform = v2.Compose([
                v2.ToImage(),
                v2.ToDtype(dtype=torch.float32, scale=True)
            ])

        def __len__(self):
            return len(self.images)

        def __getitem__(self, index):
            image = Image.open(self.images[index]).convert("RGB")
            box = get_bbox(self.bboxes[index])
            label = self.labels[index]

            image = self.transform(image)
            target = {
                "bbox": torch.tensor([box], dtype=torch.float32),
                "label": torch.tensor([label], dtype=torch.int64)
            }
            
            return image, target
        
        
    trainval_dataset = CatDogDataset()


    train_indices, val_indices = train_test_split(
        list(range(len(trainval_dataset))),
        test_size=0.2,
        stratify=[label for label in trainval_dataset.labels]
    )

    train_subset = Subset(trainval_dataset, train_indices)
    val_subset = Subset(trainval_dataset, val_indices)

    TRAIN_DATALOADER = DataLoader(train_subset, batch_size=64, shuffle=True, num_workers=0, collate_fn=lambda x: tuple(zip(*x)))
    VAL_DATALOADER = DataLoader(val_subset, batch_size=64, shuffle=False, num_workers=0, collate_fn=lambda x: tuple(zip(*x)))

    return TRAIN_DATALOADER, VAL_DATALOADER


def main():
    pass


if __name__ == "__main__":
    main()