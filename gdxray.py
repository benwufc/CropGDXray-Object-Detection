import os
import numpy as np
import torch
import torch.utils.data
from PIL import Image

class CroppedGdxrayDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir, transforms=None):
        self.root = dataset_dir
        self.transforms = transforms

    def load_boxes(self, series):
        # series: [C0001, C0002,...] is a list        
        id_format = "{series}/{series}_{id}.png"
        box_map = {}
        for s in series:
            metadata_file = os.path.join(self.root, s,"ground_truth.txt")
            if os.path.exists(metadata_file):
                for row in np.loadtxt(metadata_file):
                    row_id = int(row[0])
                    image_id = id_format.format(series=s, id=row_id)
                    box = [row[1],row[3],row[2],row[4]] # (x1, y1, x2, y2)
                    box_map.setdefault(image_id,[])
                    box_map[image_id].append(box)
        #return box_map
        self.box_map =box_map
    def __getitem__(self, idx):
        img_path, boxes = list(self.box_map.items())[idx]
        img = Image.open(os.path.join(self.root, img_path))
        
        num_objs = len(boxes)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        target = {}
        target['boxes'] = boxes
        target["labels"] = labels
        
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target
    def __len__(self):
        if not hasattr(self, 'box_map'):
            raise ValueError("You should execute Attribute 'load_boxes' first!")
        return len(self.box_map)
