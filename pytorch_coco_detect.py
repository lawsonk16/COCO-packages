import torch
import torch.utils.data
import torchvision
import json
from PIL import Image
from pycocotools.coco import COCO
import os
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from matplotlib import pyplot as plt
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import imageio
import imgaug as ia
from imgaug import augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import imageio
import imgaug as ia
from imgaug import augmenters as iaa
import torchvision.transforms.functional as TF

class myTrainDataset(torch.utils.data.Dataset):
    def __init__(self, root, annotation, transforms=None):
        self.root = root
        self.transforms = transforms
        self.coco = COCO(annotation)
        self.ids = list(sorted(self.coco.imgs.keys()))
        
    def __len__(self):
            return len(self.ids)
    def __getitem__(self, index):
        
        seq = iaa.Sequential([
            iaa.Sometimes(0.5, iaa.GaussianBlur((0, 1.5))), # apply Gaussian blur with a sigma between 0 and 3 to 50% of the images
            # apply from 0 to 3 of the augmentations from the list
            iaa.SomeOf((0, 5),[
                iaa.Sharpen(alpha=(0, 1.0), lightness=(0.15, 1.0)), # sharpen images
                iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)), # emboss images
                iaa.Fliplr(1.0), # horizontally flip
                iaa.Flipud(1.0),
                iaa.GammaContrast(gamma = (0,1.0))
            ]),
            iaa.Sometimes(0.5, iaa.AdditiveGaussianNoise(scale=(0, 25))),
        ],
        random_order=True # apply the augmentations in random order
        )
        
        # Own coco file
        coco = self.coco
        # Image ID
        img_id = self.ids[index]
        # List: get annotation id from coco
        ann_ids = coco.getAnnIds(imgIds=img_id)
        # Dictionary: target coco_annotation file for an image
        coco_annotation = coco.loadAnns(ann_ids)
        # path for input image
        path = coco.loadImgs(img_id)[0]['file_name']
        # open the input image
        image = imageio.imread(os.path.join(self.root, path), pilmode="RGB")
        
        # number of objects in the image
        num_objs = len(coco_annotation)

        # Bounding boxes for objects
        # In coco format, bbox = [xmin, ymin, width, height]
        # In pytorch, the input should be [xmin, ymin, xmax, ymax]
        bbs = []
        for i in range(num_objs):
            xmin = coco_annotation[i]['bbox'][0]
            ymin = coco_annotation[i]['bbox'][1]
            xmax = xmin + coco_annotation[i]['bbox'][2]
            ymax = ymin + coco_annotation[i]['bbox'][3]
            bbs.append(BoundingBox(x1=xmin, x2=xmax, y1=ymin, y2=ymax))

        boxes = BoundingBoxesOnImage(bbs, shape=image.shape)
        img, boxes = seq(image=image, bounding_boxes=boxes)
        
        img = TF.to_pil_image(img)

        bboxes = []
        bboxes = [[int(a.x1),int(a.y1), int(a.x2),int(a.y2)] for a in boxes.bounding_boxes]
        
        boxes = torch.as_tensor(bboxes, dtype=torch.float32)
        
        # Labels 
        labels = []
        for i in range(num_objs):
            label = coco_annotation[i]['category_id']
            labels.append(label)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        # Tensorise img_id
        img_id = torch.tensor([img_id])
        # Size of bbox (Rectangular)
        areas = []
        for i in range(num_objs):
            areas.append(coco_annotation[i]['area'])
        areas = torch.as_tensor(areas, dtype=torch.float32)
        # Iscrowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # Annotation is in dictionary format
        my_annotation = {}
        my_annotation["boxes"] = boxes
        my_annotation["labels"] = labels
        my_annotation["image_id"] = img_id
        my_annotation["area"] = areas
        my_annotation["iscrowd"] = iscrowd

        if self.transforms is not None:
            img = self.transforms(img)

        return img, my_annotation

class myTestDataset(torch.utils.data.Dataset):
    def __init__(self, root, annotation, transforms=None):
        self.root = root
        self.transforms = transforms
        self.coco = COCO(annotation)
        self.ids = list(sorted(self.coco.imgs.keys()))
        
    def __len__(self):
            return len(self.ids)
    def __getitem__(self, index):
        
        # Own coco file
        coco = self.coco
        # Image ID
        img_id = self.ids[index]
        # List: get annotation id from coco
        ann_ids = coco.getAnnIds(imgIds=img_id)
        # Dictionary: target coco_annotation file for an image
        coco_annotation = coco.loadAnns(ann_ids)
        # path for input image
        path = coco.loadImgs(img_id)[0]['file_name']
        # open the input image
        img = imageio.imread(os.path.join(self.root, path), pilmode="RGB")
     
        
        # number of objects in the image
        num_objs = len(coco_annotation)

        # Bounding boxes for objects
        # In coco format, bbox = [xmin, ymin, width, height]
        # In pytorch, the input should be [xmin, ymin, xmax, ymax]
        bbs = []
        for i in range(num_objs):
            xmin = coco_annotation[i]['bbox'][0]
            ymin = coco_annotation[i]['bbox'][1]
            xmax = xmin + coco_annotation[i]['bbox'][2]
            ymax = ymin + coco_annotation[i]['bbox'][3]
            bbs.append([xmin, ymin, xmax, ymax])
        
        boxes = torch.as_tensor(bbs, dtype=torch.float32)
        
        # Labels 
        labels = []
        for i in range(num_objs):
            label = coco_annotation[i]['category_id']
            labels.append(label)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        # Tensorise img_id
        img_id = torch.tensor([img_id])
        # Size of bbox (Rectangular)
        areas = []
        for i in range(num_objs):
            areas.append(coco_annotation[i]['area'])
        areas = torch.as_tensor(areas, dtype=torch.float32)
        # Iscrowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # Annotation is in dictionary format
        my_annotation = {}
        my_annotation["boxes"] = boxes
        my_annotation["labels"] = labels
        my_annotation["image_id"] = img_id
        my_annotation["area"] = areas
        my_annotation["iscrowd"] = iscrowd

        if self.transforms is not None:
            img = self.transforms(img)

        return img, my_annotation
    
def get_transform():
    custom_transforms = []
    custom_transforms.append(torchvision.transforms.ToTensor())
    return torchvision.transforms.Compose(custom_transforms)

# collate_fn needs for batch
def collate_fn(batch):
    return tuple(zip(*batch))

def detector_dts(model, data_loader, dt_path, iuo_nms):
    '''
    Purpose: Test a loaded model on a given data set
    TODO: make a better save file name to keep track of model/results relationships
    '''
    # Establish device settings and set model in appropriate mode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model.eval()

    detections = []
    i = 0

    # Evaluate all images
    with torch.no_grad():
        for imgs, annotations in data_loader:
            # Get data ready to evaluate
            imgs = list(img.to(device) for img in imgs)
            annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]

            # Return detections
            d = model(imgs)

            # Pull this image's image ID
            im_id = int(annotations[0]['image_id'])

            # Perform non-max suppression to remove overlapping detections
            nms_indices = torchvision.ops.nms(d[0]['boxes'], d[0]['scores'], iuo_nms).tolist()

            # Get boxes, labels, scores 
            boxes = d[0]['boxes'].tolist()
            labels = d[0]['labels'].tolist()
            scores = d[0]['scores'].tolist()

            # Only process results which survive nms
            for a in nms_indices:
                # re-factor bbox and get detection area
                bbox = boxes[a]
                new_bbox = [int(bbox[0]), int(bbox[1]), int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1])]
                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

                # Create coco-style detection
                detection = {'id': i, 'image_id': im_id,
                            'category_id': labels[a], 'score' : scores[a],
                            'bbox': new_bbox, 'area': area, 'iscrowd': 0}
                detections.append(detection)
                i += 1   

    # Save out detections
    if os.path.exists(dt_path):
        os.remove(dt_path)  
    with open(dt_path, 'w') as f:
        json.dump(detections, f) 

    return

def get_model_instance_segmentation(num_classes, pre_trained):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=pre_trained)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

def make_train_loader(data_dir, gt, batch_size, num_workers):
    '''
    Purpose: create a data loader using a coco style gt file
    '''
    # create own Dataset
    my_dataset = myTrainDataset(root = data_dir,
                              annotation = gt,
                              transforms = get_transform())

    # own DataLoader
    data_loader = torch.utils.data.DataLoader(my_dataset,
                                              batch_size = batch_size,
                                              shuffle = True,
                                              num_workers = num_workers,
                                              collate_fn = collate_fn)

    return data_loader

def make_test_loader(data_dir, gt, batch_size, num_workers):
    '''
    Purpose: create a data loader using a coco style gt file
    '''
    # create own Dataset
    my_dataset = myTestDataset(root = data_dir,
                              annotation = gt,
                              transforms = get_transform())

    # own DataLoader
    data_loader = torch.utils.data.DataLoader(my_dataset,
                                              batch_size = batch_size,
                                              shuffle = True,
                                              num_workers = num_workers,
                                              collate_fn = collate_fn)

    return data_loader

def define_model_path(save_folder, num_classes, model_name, data_name, optim, lr, mom, wd, pretrained, batch_size):
    '''
    Purpose: Create a distinctive path for your model. If a model of this type,
    with this optimizer, etc has been trained before, start on the highest epoch 
    that model has reached. Else, start at epoch 0
    IN:
      save_folder: broad folder for storing models
      num_classes: number of classes
      model_name: str, descriing architecture
                  actual saved folder will be save_folder/model_name/path
      data_name: str, describing data used
      optim: string, describing your optimizer type
      lr, mom, wd: floats, learning rate/momentum/weight decay for optimizer
      pretrained: bool, whether model is pretrained
      batch_size: int, batch size
    OUT:
      path to model
    '''

    # Open the correct folder, creating it if necessry
    model_folder = save_folder + model_name + '/'
    if not os.path.exists(model_folder):
        os.mkdir(model_folder)

    # Get a list of all current models
    current_models = os.listdir(model_folder)

    # List all relevant parameters, in a fixed order after key id strings
    path_params = [data_name, 'classes', num_classes, 
                          'optim', optim, 'lr', lr, 'mom', mom, 'wd', wd,
                          'pretrained', pretrained,
                          'batch', batch_size, 'epochs']

    # Ensure all params are saved as strings, then create a key to id this model config
    path_params = [str(p) for p in path_params]
    model_path = '_'.join(path_params).replace('.', 'p')

    # Figure out if this config has already been used
    start_epoch = 0

    for m in current_models:
        if model_path in m and '.pt' in m:
            # If some training has already occurred in this config, start off with that saved model
            epoch = int(m.split('_')[-1].replace('.pt', ''))
            if epoch > start_epoch:
                start_epoch = epoch

    # Save definitive model path
    path = model_folder + model_path + '_' + str(start_epoch) + '.pt'

    return path

def train_one_epoch(model, data_loader, optimizer, device, path):
    model.train()
    len_dataloader = len(data_loader)
    i = 0  
    # Process all data in the data loader  
    for imgs, annotations in data_loader:
        
        # Prepare images and annotations
        imgs = list(img.to(device) for img in imgs)
        annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
        
        # Calculate loss and backpropagate
        loss_dict = model(imgs, annotations)
        losses = sum(loss for loss in loss_dict.values())
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        # Provide feedback
        if i % 25 == 0:
            print(f'Iteration: {i}/{len_dataloader}, Loss: {losses}')

        # Save model out occasionally
        if i % 50 == 0:
           torch.save(model.state_dict(), path) 

        i += 1
    # Upon completion of the whole epoch
    ep = int(path.split('_')[-1].replace('.pt', ''))
    path = '_'.join(path.split('_')[:-1]) + '_' + str(ep + 1) + '.pt'
    torch.save(model.state_dict(), path)
    return model, path, optimizer

def detector_dts(model, data_loader, dt_path, iuo_nms):
    '''
    Purpose: Test a loaded model on a given data set
    TODO: make a better save file name to keep track of model/results relationships
    '''
    # Establish device settings and set model in appropriate mode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model.eval()

    detections = []
    i = 0

    # Evaluate all images
    with torch.no_grad():
        for imgs, annotations in data_loader:
            # Get data ready to evaluate
            imgs = list(img.to(device) for img in imgs)
            annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]

            # Return detections
            d = model(imgs)

            # Pull this image's image ID
            im_id = int(annotations[0]['image_id'])

            # Perform non-max suppression to remove overlapping detections
            nms_indices = torchvision.ops.nms(d[0]['boxes'], d[0]['scores'], iuo_nms).tolist()

            # Get boxes, labels, scores 
            boxes = d[0]['boxes'].tolist()
            labels = d[0]['labels'].tolist()
            scores = d[0]['scores'].tolist()

            # Only process results which survive nms
            for a in nms_indices:
                # re-factor bbox and get detection area
                bbox = boxes[a]
                new_bbox = [int(bbox[0]), int(bbox[1]), int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1])]
                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

                # Create coco-style detection
                detection = {'id': i, 'image_id': im_id,
                            'category_id': labels[a], 'score' : scores[a],
                            'bbox': new_bbox, 'area': area, 'iscrowd': 0}
                detections.append(detection)
                i += 1   

    # Save out detections
    if os.path.exists(dt_path):
        os.remove(dt_path)  
    with open(dt_path, 'w') as f:
        json.dump(detections, f) 

    return