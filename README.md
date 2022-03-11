The following is necessary steps in this object detection task.

# Dataset

```
import gdxray

# use our dataset and defined transformations
dataset = gdxray.CroppedGdxrayDataset('cropped_castings_128_256', get_transform(train=True))
dataset.load_boxes(['C0001', 'C0002'])
```

# Data augmentation

```
import transforms as T

def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)
```

# Model

```
import torchvision

def get_instance_model(num_classes):
    # load an instance segmentation model pre-trained on COCO
    #model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model
    
 # get the model using our helper function
model = get_instance_model(num_classes)
```

# Train

```
from engine import train_one_epoch

train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
```

# Evaluation

```
from engine import train_one_epoch

APs = evaluate_compute_batch_ap(model, data_loader_test, device=device)
```

# Visualization

```
import visualize

visualize.display_instances(img, target['boxes'], figsize=(8, 8))
```

