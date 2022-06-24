import torch
import torch.utils.data
import torchvision
import json
from PIL import Image
from pycocotools.coco import COCO
import os
from tqdm import tqdm
import numpy as np
import copy
import random

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
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.rpn import AnchorGenerator

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

def detector_dts(model, model_path, data_loader, iou_nms, data_split):
    '''
    Purpose: Test a loaded model on a given data set
    TODO: make a better save file name to keep track of model/results relationships
    '''
    # Establish device settings and set model in appropriate mode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model.to(device)
    model.eval()

    # Create a named path for your detections
    dt_path = name_dts(model_path, data_split, iou_nms)

    if os.path.exists(dt_path):
        print('Detections already produced')
        return dt_path

    detections = []
    i = 0

    # Evaluate all images
    with torch.no_grad():
        for imgs, annotations in tqdm(data_loader, desc = f'Getting {data_split} detections'):
            # Get data ready to evaluate
            imgs = list(img.to(device) for img in imgs)
            annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]

            # Return detections
            d = model(imgs)

            # Pull this image's image ID
            im_id = int(annotations[0]['image_id'])

            # Perform non-max suppression to remove overlapping detections
            nms_indices = torchvision.ops.nms(d[0]['boxes'], d[0]['scores'], iou_nms).tolist()

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

    return dt_path

def get_fasterrcnn(num_classes, pre_trained, resnet_depth = 50):
    try:
        backbone = resnet_fpn_backbone(f'resnet{resnet_depth}', pre_trained)
        model = FasterRCNN(backbone, num_classes)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        
        return model
    except:
        print('This model depth is not supported. Please make sure the model depth is a supported integer value')
        return


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
    # Process all data in the data loader 
    epoch_losses = [] 
    for imgs, annotations in tqdm(data_loader):
        
        # Prepare images and annotations
        imgs = list(img.to(device) for img in imgs)
        annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
        
        # Calculate loss and backpropagate
        loss_dict = model(imgs, annotations)
        losses = sum(loss for loss in loss_dict.values())
        epoch_losses.append(losses)
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

    # Upon completion of the whole epoch
    epoch_loss = np.mean(epoch_losses)
    print(f'Epoch loss: {epoch_loss}')
    ep = int(path.split('_')[-1].replace('.pt', ''))
    path = '_'.join(path.split('_')[:-1]) + '_' + str(ep + 1) + '.pt'
    torch.save(model.state_dict(), path)
    return model, path, optimizer


def train_fasterrcnn(model, model_path, data_loaders, optim, lr, mom, wd, epochs, start_eval_epoch, eval_freq, iou_nms, save_freq):
    
    ### Initialization ###
    # Defining data structures to store train and test info 
    losses_train = []
    accs_train = []
    losses_val = []
    accs_val = []
    best_model_weights = model.state_dict()
    current_epoch = 0

    # set to monitor performance
    lowest_val_loss = 9999999999999999999
    lowest_train_loss = 9999999999999999999
    
    # load any previous training
    if os.path.exists(model_path):
        # load model information
        model_info = torch.load(model_path)
        # weights
        best_model_weights = model_info['weights']
        model.load_state_dict(best_model_weights)
        # epoch
        current_epoch = model_info['epochs_trained']
        # loss
        losses_train = model_info['losses train']
        losses_val = model_info['losses val']
        # accuracy
        accs_train = model_info['acc train']
        accs_val = model_info['acc val']
        # lowest train loss
        lowest_val_loss = min(losses_val)
        lowest_train_loss = min(losses_train)

        print(f'Loading {current_epoch} epochs of training for fasterrcnn')
    
    # Define path to save out models over time
    historic_weights = '/'.join(model_path.split('/')[:-1]) + '/historic_weights/'

    if not os.path.exists(historic_weights):
        os.mkdir(historic_weights)

    # unpack data loaders
    train_loader, val_loader = data_loaders

    # find your device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model.to(device)

    # parameters
    params = [p for p in model.parameters() if p.requires_grad]

    # Set optimizer
    if optim == 'SGD':
        optimizer = torch.optim.SGD(params, lr=lr, momentum=mom, weight_decay=wd)
    elif optim == 'Adam':
        optimizer = torch.optim.Adam(params, lr=lr)

    for epoch in range(current_epoch, epochs):
        
        print(f'Training epoch {epoch + 1} of {epochs}')
        ### Training ###
        model.train()
        epoch_train_losses = []
        # Process all data in the data loader 
        for imgs, annotations in tqdm(train_loader, desc = 'Training'):
            
            # Prepare images and annotations
            imgs = list(img.to(device) for img in imgs)
            annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
            
            # Calculate loss 
            loss_dict = model(imgs, annotations)
            losses = sum(loss for loss in loss_dict.values())
            epoch_train_losses.append(losses.cpu().detach().numpy())

            # Backprop
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
        
        # Train epoch done
        epoch_train_loss = np.mean(epoch_train_losses)
        losses_train.append(epoch_train_loss)
        
        ### Validation ###
        epoch_val_losses = []
        # Process all data in the data loader 
        for imgs, annotations in tqdm(val_loader, desc = 'Validation'):
            
            # Prepare images and annotations
            imgs = list(img.to(device) for img in imgs)
            annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
            
            # Calculate loss 
            with torch.no_grad():
                loss_dict = model(imgs, annotations)
            losses = sum(loss for loss in loss_dict.values())
            epoch_val_losses.append(losses.cpu().detach().numpy())

        # Val epoch done
        epoch_val_loss = np.mean(epoch_val_losses)
        losses_val.append(epoch_val_loss)

        ### Evaluation and historic model saving ###
        if epoch >= start_eval_epoch:
            if epoch % eval_freq == (eval_freq-1):
                hist_model_path = historic_weights + f'{epoch+1}_epochs'

                print(f'Evalutation on epoch {epoch + 1}')
                # val dts
                detector_dts(model, model_path, val_loader, iou_nms, 'val')
        
        ### Save Out ###
        if epoch_val_loss < lowest_val_loss:
            best_model_weights = copy.deepcopy(model.state_dict())
            lowest_train_loss = epoch_train_loss
            print(f'Lowest val loss: {lowest_val_loss}')

        # either way, save the model
        model_info = {'weights': best_model_weights,
                      'epochs_trained': epoch + 1,
                      'least val loss': lowest_val_loss,
                      'acc train': accs_train,
                      'acc val': accs_val,
                      'losses train': losses_train,
                      'losses val': losses_val}
        # If this is an evaluation epoch, also save out the model
        if epoch > 0:
            if epoch % save_freq == (save_freq-1):
                hist_model_path = historic_weights + f'{epoch+1}_epochs'
                torch.save(model_info, hist_model_path)
                print('Saved model history')
        torch.save(model_info, model_path)

    return losses_train

def name_model(exp_folder, data_name, data_split, resnet_backbone, batch_size, num_classes, optim, lr, mom, wd, pretrained):
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

    REPLACES: define_model_path
    '''

    # Name a folder for experiments with this dataset, creating it if necessary
    data_folder = exp_folder + data_name + '/'
    if not os.path.exists(data_folder):
        os.mkdir(data_folder)

    # Add a folder for this data split, creating it if necessary
    split_folder = data_folder + data_split + '/'
    if not os.path.exists(split_folder):
        os.mkdir(split_folder)

    # Add a folder for this kind of resnet backbone, creating it if necessary
    model_name = f'resnet{resnet_backbone}fpn'
    res_folder = split_folder + model_name + '/'
    if not os.path.exists(res_folder):
        os.mkdir(res_folder)

    # List all relevant parameters, in a fixed order after key id strings
    path_params = ['classes', num_classes, 
                   'optim', optim, 'lr', lr, 'mom', mom, 'wd', wd,
                   'pretrained', pretrained,
                   'batch', batch_size]

    # Ensure all params are saved as strings, then create a key to id this model config
    path_params = [str(p) for p in path_params]
    model_path = '_'.join(path_params).replace('.', 'p')

    # Make Experiment Folder
    exp_folder = res_folder + model_path + '/' 

    if not os.path.exists(exp_folder):
        os.mkdir(exp_folder)

    path = exp_folder + 'best_weights.pt'

    return path

def name_dts(model_path, data_split, iou_nms):
    '''
    Goal: Name a set of detections
    '''
    # load model information and model
    if os.path.exists(model_path):
        # load model information
        model_info = torch.load(model_path)
        # weights
        best_model_weights = model_info['weights']
        # epoch
        starting_epoch = model_info['epochs_trained']

    # define the path to the detection files for this model
    dts_folder = '/'.join(model_path.split('/')[:-1]) + '/detections/'

    if not os.path.exists(dts_folder):
        os.mkdir(dts_folder)

    iou_nms_str = str(iou_nms).replace('.','p')

    dts_path = dts_folder + data_split + f'_{starting_epoch+1}_epochs_{iou_nms_str}_iou-nms.json'
    
    return dts_path

def load_model(model, model_path):
    
    ### Initialization ###
    
    # load any previous training
    if os.path.exists(model_path):
        # load model information
        model_info = torch.load(model_path)
        # weights
        best_model_weights = model_info['weights']
        model.load_state_dict(best_model_weights)
    
    return model

def get_most_recent_dts(model_path):
    dt_path = '/'.join(model_path.split('/')[:-1]) + '/detections/'
    all_dts = os.listdir(dt_path)
    highest_epoch = 0
    train_dts = ''
    val_dts = ''
    for dt in all_dts:
        epoch = int(dt.split('_')[1])
        if epoch > highest_epoch:
            highest_epoch = epoch
            if 'train' in dt:
                train_dts = dt
                val_dts = dt.replace('train', 'val')
            else:
                train_dts = dt.replace('val', 'train')
                val_dts = dt
    return dt_path + train_dts, dt_path + val_dts