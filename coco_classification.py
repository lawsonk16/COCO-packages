import os
import json
from tqdm import tqdm
from matplotlib import pyplot as plt


def anns_on_image(im_id, annotations):
    '''
    IN: 
        - im_id: int id for 'id' in 'images' of coco json
        - json_path: path to coco gt json
    OUT:
        - on_image: list of annotations on the given image
    '''
    
    # Create list of anns on this image
    on_image = []
    for a in annotations:
        if a['image_id'] == im_id:
            on_image.append(a)
    
    return on_image

def get_category_gt(i, categories):
    '''
    IN: 
        -i: int of 'category_id' you would like identified
        -categories: 'categories' section of coco json
    OUT: 
        -name of object category, or "none" if the category isn't present
    '''
    
    for c in categories:
        if c['id'] == i:
            return c['name']
    return "None"

def classification_from_json(json_path, image_folder, classification_folder, pad=0, gsd_thresh = None):

    if not os.path.exists(classification_folder):
        os.mkdir(classification_folder)

    class_counts = {}

    with open(json_path, 'r') as f:
        contents = json.load(f)
    for cat in contents['categories']:
        class_counts[cat['name']] = 0
        class_folder = classification_folder + cat['name'] + '/'
        if not os.path.exists(class_folder):
            os.mkdir(class_folder)

    # List images
    images = contents['images']
    annotations = contents['annotations']
    categories = contents['categories']
    mistakes = 0
        
    # Process each image
    for im in tqdm(images):
        i = im['id']
        im_name = im['file_name']
        gsd = im['gsd']

        if gsd_thresh != None:
            if gsd != None:
                if gsd < gsd_thresh:
                    # Get annotations on this image
                    anns = anns_on_image(i, annotations)
                    
                    try:
                        a = anns[-1]
                        cat = a['category_id']
                        cat_name = get_category_gt(cat, categories)
                        chip_path = classification_folder + cat_name + '/' + im_name.split('_')[0].split('.')[0] + '_' + str(class_counts[cat_name]) + '.png'
                        if not os.path.exists(chip_path):   
                            # Read the image
                            im_path = image_folder + im_name
                            img = plt.imread(im_path)
                            for a in anns:
                                cat = a['category_id']
                                cat_name = get_category_gt(cat, categories)
                                chip_path = classification_folder + cat_name + '/' + im_name.split('_')[0].split('.')[0] + '_' + str(class_counts[cat_name]) + '.png'
                                class_counts[cat_name] += 1
                                if not os.path.exists(chip_path):
                                    b = a['bbox']
                                    b = [int(a) for a in b]
                                    try:
                                        chip = img[(b[1] - pad):(b[1] + b[3] + pad), (b[0] - pad):(b[0]+b[2] + pad)]
                                        plt.imsave(chip_path, chip)
                                    except:
                                        mistakes += 1
                                        continue
                        else:
                            for a in anns:
                                cat = a['category_id']
                                cat_name = get_category_gt(cat, categories)
                                class_counts[cat_name] += 1
                    except:
                        continue  
        else:
            # Get annotations on this image
            anns = anns_on_image(i, annotations)
            
            try:
                a = anns[-1]
                cat = a['category_id']
                cat_name = get_category_gt(cat, categories)
                chip_path = classification_folder + cat_name + '/' + im_name.split('_')[0].split('.')[0] + '_' + str(class_counts[cat_name]) + '.png'
                if not os.path.exists(chip_path):   
                    # Read the image
                    im_path = image_folder + im_name
                    img = plt.imread(im_path)
                    for a in anns:
                        cat = a['category_id']
                        cat_name = get_category_gt(cat, categories)
                        chip_path = classification_folder + cat_name + '/' + im_name.split('_')[0].split('.')[0] + '_' + str(class_counts[cat_name]) + '.png'
                        class_counts[cat_name] += 1
                        if not os.path.exists(chip_path):
                            b = a['bbox']
                            b = [int(a) for a in b]
                            try:
                                chip = img[(b[1] - pad):(b[1] + b[3] + pad), (b[0] - pad):(b[0]+b[2] + pad)]
                                plt.imsave(chip_path, chip)
                            except:
                                mistakes += 1
                                continue
                else:
                    for a in anns:
                        cat = a['category_id']
                        cat_name = get_category_gt(cat, categories)
                        class_counts[cat_name] += 1  
                
            except:
                continue
    return

