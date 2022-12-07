import json
import os
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import random
import json
from PIL import Image
import shutil
from tqdm import tqdm
import numpy as np
import torch

def add_gsd_to_chips(full_anns, chip_anns):

    '''
    PURPOSE: Given a set of full image annotations with gsd values and chipped
    annotations without them, add the gsd values to the corresponding chipped
    annotations for experimental use
    IN:
     - full_anns: str, 
     - chip_anns: str,
    OUT:
     - chip_anns_gsd: str,
    '''

    with open(full_anns, 'r') as f:
        data_full = json.load(f)
    images_full = data_full['images']

    with open(chip_anns, 'r') as f:
        data_chip = json.load(f)
    images_chip = data_chip['images']

    new_images_c = []

    # process each chipped image one by one
    for i_c in tqdm(images_chip, desc = 'Adding GSD to Images'):
        # copy the data
        new_i_c = i_c.copy()

        # get the full image info from the chip name
        im_name_c = i_c['file_name']

        full_im_id = int(im_name_c.split('_')[1])

        # add gsd
        new_i_c['gsd'] = get_im_gsd_from_id(full_im_id, data_full)
        new_images_c.append(new_i_c)

    data_chip['images'] = new_images_c

    chip_anns_gsd = chip_anns.split('.')[0] + '_gsd.json'

    if os.path.exists(chip_anns_gsd):
        os.remove(chip_anns_gsd)

    with open(chip_anns_gsd, 'w') as f:
        json.dump(data_chip, f)

    return chip_anns_gsd

def anns_on_image(im_id, contents):
    '''
    IN: 
        - im_id: int id for 'id' in 'images' of coco json
        - json_path: path to coco gt json
    OUT:
        - on_image: list of annotations on the given image
    '''
    
    # Pull out annotations
    anns = contents['annotations']
    
    # Create list of anns on this image
    on_image = []
    for a in anns:
        if a['image_id'] == im_id:
            on_image.append(a)
    
    return on_image

def anns_on_image_dt(im_id, json_path):
    '''
    IN: 
        - im_id: int id for 'id' in 'images' of coco json
        - json_path: path to coco gt json
    OUT:
        - on_image: list of annotations on the given image
    '''
    # Open json
    with open(json_path, 'r') as f:
        contents = json.load(f)
    
    # Create list of anns on this image
    on_image = []
    for a in contents:
        if a['image_id'] == im_id:
            on_image.append(a)
    
    return on_image

def average_bboxes_from_centerpoints(anns_path, avg_img_gsd = None):
    '''
    PURPOSE: After finding average object sizes, and using bounding boxes to add
             centerpoints (all to a coco annotation file), replace bounding boxes 
             using image GSD and average object sizes to grow the centerpoints
    IN:
     - anns_path: str, path to coco annotations file
     - avg_img_gsd: float or int, optional, average image size in dataset, 
                    which will be used as a default if an image doesn't have 
                    a noted GSD. 
    OUT:
     - new_anns_path: str, path to new annotation file
    '''
    
    # open annotation file
    with open(anns_path, 'r') as f:
        content = json.load(f)

    # if necessary, get average gsd
    if avg_img_gsd == None:
        avg_img_gsd = get_average_image_gsd(anns_path)
    
    # pull out key sections of file
    anns = content['annotations']

    # adjust bounding boxes based on centerpoints and object sizes
    new_annotations = []
    for a in tqdm(anns, desc = 'Creating Square Bboxes'):
        new_a = a.copy()
        [x,y] = a['centerpoint']
        obj_size = get_obj_size_from_id(a['category_id'], content)
        im_gsd = get_im_gsd_from_id(a['image_id'], content)
        if im_gsd != None:
            ob_h_w = int(obj_size/im_gsd)
        else:
            ob_h_w = int(obj_size/avg_img_gsd)
        square_bbox = [x - (ob_h_w/2), y - (ob_h_w/2), ob_h_w, ob_h_w]
        new_a['bbox'] = square_bbox
        new_annotations.append(new_a)

    content['annotations'] = new_annotations

    new_anns_path = anns_path.split('.')[0] + '_square.json'

    if os.path.exists(new_anns_path):
        os.remove(new_anns_path)
    
    with open(new_anns_path, 'w') as f:
        json.dump(content, f)

    return new_anns_path

def check_for_parity(train_gt_path, test_gt_path, chip_dir):
    '''
    Purpose: determine if all the images in a defined train/test split
    exist within a specified folder
    '''
    # Get a list of all chips in the split
    test_ims = json_images(test_gt_path)
    train_ims = json_images(train_gt_path)

    former_ims = test_ims.copy()
    former_ims.extend(train_ims)

    # List of images in the folder
    new_ims = os.listdir(chip_dir)

    # Check if all images in split are in folder
    # And all ims in folder are in split
    they_match = True

    for i in new_ims:
        if i not in former_ims:
            they_match = False

    for i in former_ims:
        if i not in new_ims:
            they_match = False

    # Print results
    if they_match:
        print("Images in these jsons and the specified folder match")
    else:
        print("These images don't quite match. Sad.")
                     
    return they_match

def choose_random_ims(num_ims, contents):
    '''
    IN:
        -num_ims: int number of image ids desired
        -contents: coco json contents
    OUT:
        -list of num_ims random image ids from the input json
    '''
    
    # Pull out key section
    images = contents['images']
    
    # Get a list of all image ids in the json
    all_ims = []
    for i in images:
        all_ims.append(i['id'])
    
    # Enusre there are no duplicates in the list
    all_ims = list(set(all_ims))
    
    # Shuffle the list
    random.shuffle(all_ims)
    
    # Create a smaller list of the num_ims requested
    rand_ims = all_ims[:num_ims]
    
    return rand_ims

def classification_from_json(json_path, image_folder, classification_folder):

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
        
    # Process each image
    for im in tqdm(images):
        i = im['id']
        im_name = im['file_name']
        
        # Get annotations on this image
        anns = anns_on_image(i, contents)
        
        # Read the image
        im_path = image_folder + im_name
        img = plt.imread(im_path)
        for a in anns:
            b = a['bbox']
            cat = a['category_id']
            cat_name = get_category_gt(cat, contents)
            chip = img[b[1]:b[1] + b[3], b[0]:b[0]+b[2]]
            chip_path = classification_folder + cat_name + '/' + im_name.split('_')[0] + '_' + str(class_counts[cat_name]) + '.png'
            class_counts[cat_name] += 1
            if not os.path.exists(chip_path):
                plt.imsave(chip_path, chip)
    return

def convert_anns_centerpoint(anns_path, max_shift = 5):
    '''
    PURPOSE: Convert an annotation file with image-oriented bounding boxes to 
             center point annotations instead
    IN:
     - anns_path: str, path to annotations
     - max_shift: int, maximum distance the point may shift 
    OUT:
     - new_anns_path: str, path to new annotations
    '''

    # open the annotation file
    with open(anns_path, 'r') as f:
        ann_contents = json.load(f)
    
    # grab those annotations
    annotations = ann_contents['annotations']

    # add randomly shifted centerpoints to each annotation
    new_anns = []
    for a in annotations:
        new_a = a.copy()
        x1, y1, w, h = a['bbox']
        x_c = x1 + int(w/2)
        y_c = y1 + int(h/2)

        new_a['centerpoint'] = random_shift_point([x_c, y_c], max_shift)
        new_anns.append(new_a)

    ann_contents['annotations'] = new_anns

    # create and save new annotation file
    new_anns_path = anns_path.split('.')[0] + f'_cp_{max_shift}.json'

    if os.path.exists(new_anns_path):
        os.remove(new_anns_path)
    
    with open(new_anns_path, 'w') as f:
        json.dump(ann_contents, f)

    return new_anns_path

def convert_imgs_rgb(folder):
    '''
    Purpose: force all images in folder to assume kosher image format
    '''
    test_images = os.listdir(folder)
    test_images = [folder + i for i in test_images]
    count  = 0
    for i in test_images:   
        img = Image.open(i).convert('RGB')
        img.save(i)
        count += 1
        if count % 250 == 0:
            print(count)
    
    return

def display_random_ims(num_ims, json_path, image_folder, fig_size = (20,20), text_on = True):
    '''
    PURPOSE: Display some number of images from a coco dataset, randomly selected
    IN:
        -num_ims: int indicating how many to display
        -json_path: coco gt file
        -image_folder: folder where images in json_path are located
    OUT:
        -figures with each randomly selected image and its annotations
    
    '''
    
    # open json at the start of the process
    with open(json_path, 'r') as f:
        gt = json.load(f)

    # Get Color palette
    pal = make_palette(gt)
    
    # Pick the image ids to display
    ims = choose_random_ims(num_ims, gt)

    # Process each image
    for i in ims:
        
        images = gt['images']
        for im in images:
            if im['id'] == i:
                im_name = im['file_name']

        # Get annotations on this image
        anns = anns_on_image(i, gt)
        
        # Display the image
        im_path = image_folder + im_name
        plt.figure()
        f,ax = plt.subplots(1, figsize = fig_size)
        img = plt.imread(im_path)
        channels = len(img.shape)
        plt.imshow(img)
        for a in anns:
            b = a['bbox']
            cat = a['category_id']
            cat_name = get_category_gt(cat, gt)
            rect = patches.Rectangle((b[0], b[1]), b[2], b[3], edgecolor = pal[cat-1], facecolor = "none")
            if text_on:
                plt.text(b[0], b[1], cat_name, ha = "left", color = 'w')
            ax.add_patch(rect)
        plt.title(im_name)
        plt.show()
    
    return


def display_random_dt(num_ims, gt_path, dt_path, image_folder, fig_size = (20,20), conf_thresh = 0.9):
    '''
    PURPOSE: Display some number of images from a coco dataset, randomly selected
    IN:
        -num_ims: int indicating how many to display
        -json_path: coco gt file
        -image_folder: folder where images in json_path are located
    OUT:
        -figures with each randomly selected image and its annotations
    '''
    with open(gt_path, 'r') as f:
        gt_content = json.load(f)

    # Get Color palette
    pal = make_palette(gt_content)
    
    # Pick the image ids to display
    ims = choose_random_ims(num_ims, gt_content)
    
    with open(gt_path, 'r') as f:
        gt = json.load(f)
        
    # Process each image
    for i in ims:
        
        images = gt['images']
        for im in images:
            if im['id'] == i:
                im_name = im['file_name']
        
        # Get annotations on this image
        anns_dt = anns_on_image_dt(i, dt_path)
        
        # Display the image
        im_path = image_folder + im_name
        plt.figure()
        f,ax = plt.subplots(1, figsize = fig_size)
        img = plt.imread(im_path)
        plt.imshow(img)
        for a in anns_dt:
            b = a['bbox']
            cat = a['category_id']
            conf = a['score']
            if conf >= conf_thresh:
                cat_name = get_category_gt(cat, gt)
                rect = patches.Rectangle((b[0], b[1]), b[2], b[3], edgecolor = pal[cat-1], facecolor = "none", ls = '--')
                plt.text(b[0], b[1], cat_name, ha = "left", color = 'w')
                ax.add_patch(rect)
        plt.show()
    
    return  

def display_random_gt_dt(num_ims, gt_path, dt_path, image_folder, fig_size = (20,20), conf_thresh = 0.9):
    '''
    PURPOSE: Display some number of images from a coco dataset, randomly selected
    IN:
        -num_ims: int indicating how many to display
        -json_path: coco gt file
        -image_folder: folder where images in json_path are located
    OUT:
        -figures with each randomly selected image and its annotations
    '''

    with open(gt_path, 'r') as f:
        gt_content = json.load(f)

    # Get Color palette
    pal = make_palette(gt_content)
    
    # Pick the image ids to display
    ims = choose_random_ims(num_ims, gt_content)
    
    with open(gt_path, 'r') as f:
        gt = json.load(f)
        
    # Process each image
    for i in ims:
        
        images = gt['images']
        for im in images:
            if im['id'] == i:
                im_name = im['file_name']
        
        # Get annotations on this image
        anns_dt = anns_on_image_dt(i, dt_path)
        anns_gt = anns_on_image(i, gt_content)
        
        # Display the image
        im_path = image_folder + im_name
        plt.figure()
        f,ax = plt.subplots(1, figsize = fig_size)
        img = plt.imread(im_path)
        plt.imshow(img)
        for a in anns_dt:
            b = a['bbox']
            cat = a['category_id']
            conf = a['score']
            if conf >= conf_thresh:
                cat_name = get_category_gt(cat, gt)
                rect = patches.Rectangle((b[0], b[1]), b[2], b[3], edgecolor = pal[cat-1], facecolor = "none", ls = '--')
                plt.text(b[0], b[1], cat_name, ha = "left", color = 'w')
                ax.add_patch(rect)
        for a in anns_gt:
            b = a['bbox']
            cat = a['category_id']
            cat_name = get_category_gt(cat, gt)
            rect = patches.Rectangle((b[0], b[1]), b[2], b[3], edgecolor = pal[cat-1], facecolor = "none", ls = '-')
            plt.text(b[0], b[1], cat_name, ha = "right", color = 'b')
            ax.add_patch(rect)
        plt.show()
    
    return

def estimate_category_size(anns_path, write_out = False, matched_files = []):
    '''
    PURPOSE: Get average sizes in meters for each object category in a 
             coco dataset and optionally add them to the file, with the option
             to add the values to multiple files
    IN:
     - anns_path: str, path to coco annotation file
     - write_out: boolean, whether or not to write the values into the coco file
     - matched_files : list of strs, paths to other files to write our average 
                       object sizes to
    OUT:
     - estimates: dict, contains information about each category keyed to its id
    '''

    # open annotation file
    with open(anns_path, 'r') as f:
        content = json.load(f)
    
    # pull out key sections of file
    cats = content['categories']
    anns = content['annotations']

    # create dictionary for storing key info about objct sizes
    estimates = {}
    for c in cats:
        estimates[c['id']] = {'name': c['name'], 'sizes': []}

    # add the size of each indivdual object to the list for that category
    for a in tqdm(anns):
        bbox = a['bbox']

        # get largest side of object
        size = max(bbox[2:3])
        im_gsd = get_im_gsd_from_id(a['image_id'], content)
        if im_gsd != None:
          size_m = size*im_gsd
        estimates[a['category_id']]['sizes'].append(size_m)
    
    # add average sizes using size lists
    for k,v in estimates.items():
        avg = np.mean(v['sizes'])
        name = v['name']
        estimates[k]['average'] = avg

    new_cats = []
    if write_out:

        for c in cats:
            new_c = c.copy()
            new_c['average_size'] = estimates[c['id']]['average']
            new_cats.append(new_c)
        content['categories'] = new_cats

        os.remove(anns_path)

        with open(anns_path, 'w') as f:
            json.dump(content, f)
        
        for fp in matched_files:
            with open(fp, 'r') as f:
                f_contents = json.load(f)
            f_contents['categories'] = new_cats
            os.remove(fp)
            with open(fp, 'w') as f:
                json.dump(f_contents, f)
    return estimates

def experiment_from_percentage(percentage, img_dir, new_img_dir, gt, new_gt_path):
    '''
    IN: 
       - percentage: int, representing the percentage (out of 100) 
                     of the images you would like to keep
       - img_dir: str, image directory
       - new_img_dir: str, location you'd like the new set of images
       - gt: str, path to coco formatted json ground truth
       - new_gt: str, path to new gt location
    OUT: a new experiment ready to go in the specified locations
    PURPOSE: Use to create a new experimental directroy with a % of your data for 
    faster testing of other procedures on a given dataset
    '''
    # Get a list of new images
    ims_percentage = get_im_list_percentage(percentage, img_dir)

    # Copy the list to the new location
    for f in ims_percentage:
        src = img_dir + f
        dst = new_img_dir + f
        shutil.copy2(src, dst)

    # Make a new gt file
    gt_from_im_list(gt, ims_percentage, new_gt_path)

    return

def get_anns_in_box(box, anns):
    b_anns = []
    
    # Get image coordinates
    i_x1 = box[0]
    i_y1 = box[1]
    i_x2 = i_x1 + box[2]
    i_y2 = i_y1 + box[3]
    
    # Check each individual annotation
    for a in anns:
        b = a['bbox']
        x1 = b[0]
        y1 = b[1]
        x2 = x1 + b[2]
        y2 = y1 + b[3]
        
        # Annotations will be assigned by centerpoint
        xc = (x1 + x2)/2
        yc = (y1 + y2)/2
        
        # Check if centerpoint is within chip
        if xc > i_x1 and xc < i_x2 and yc > i_y1 and yc < i_y2:
            # adjust coordinates to this chip
            n_x1 = x1 - i_x1
            n_y1 = y1 - i_y1
            n_x2 = n_x1 + b[2]
            n_y2 = n_y1 + b[3]
            
            # Ensure this new annotation is fully on-chip
            if n_x1 < 0:
                n_x1 = 0
            if n_y1 < 0:
                n_y1 = 0
            if n_x2 > i_x2:
                n_x2 = i_x2
            if n_y2 > i_y2:
                n_y2 = i_y2
            
            # Calculate new width and height after key checks
            n_w = n_x2 - n_x1
            n_h = n_y2 - n_y1
            
            new_a = a.copy()
            new_a['bbox'] = [n_x1, n_y1, n_w, n_h]
            
            b_anns.append(new_a)
    return b_anns 

def get_average_image_gsd(anns_path):
    '''
    PURPOSE: Find the average GSD of the images in a coco ground truth file
    IN:
     - anns_path: str, path to coco annotation file
    OUT:
     - avg_img_gsd: float, average gsd of images in dataset
    '''
    with open(anns_path, 'r') as f:
        content = json.load(f)
    images = content['images']

    gsd_vals = []

    for i in images:
        if i['gsd'] != None:
            gsd_vals.append(i['gsd'])

    avg_img_gsd = np.average(gsd_vals)

    return avg_img_gsd

def get_category(i, categories):
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

def get_category_counts(json_path):
    '''
    IN: json_path: path to coco json gt file
    OUT: dict of form {category name: count of objects}
    '''
    # Open json file
    with open(json_path, 'r') as f:
        contents = json.load(f)
    
    # Pull out key sections
    anns = contents['annotations']
    cats = contents['categories']

    # Create dictionary of by-class counts
    cat_counts = {}
    for a in anns:
        cat = a['category_id']
        name = get_category(cat, cats)
        if name not in cat_counts:
            cat_counts[name] = 1
        else:
            cat_counts[name] = cat_counts[name] + 1
    
    return cat_counts

def get_category_id_from_name(cat_name, gt_content):
    '''
    IN: 
      - cat_name: str, category name from coco json
      - gt_content: json contents of coco gt file
    OUT: cat_id: int, category id for that named category
    '''

    for c in gt_content['categories']:
        if c['name'] == cat_name:
            return c['id']
    return None

def get_category_gt(i, gt):
    '''
    IN: 
        -i: int of 'category_id' you would like identified
        -categories: 'categories' section of coco json
        - gt: loaded coco gt information
    OUT: 
        -name of object category, or "none" if the category isn't present
    '''
        
    categories = gt['categories']
    
    for c in categories:
        if c['id'] == i:
            return c['name']
    return "None" 

def get_image_paths(image_folder):
    # Image paths
    folders = os.listdir(image_folder)
    folders = [image_folder + f + '/' for f in folders]
    image_paths = []
    for f in folders:
        images = os.listdir(f)
        images = [f + i for i in images]
        image_paths.extend(images)
    
    return image_paths

def get_im_gsd_from_id(im_id, gt_content):
    '''
    PURPOSE: Get the GSD of an image based on its id in a coco file
    IN:
     - im_id: int, image id for the image in question
     - gt_content: the content from a coco ground truth file
    OUT:
     - pt: either the image's GSD or None if it isn'available
    '''
    images = gt_content['images']

    for i in images:
        if i['id'] == im_id:
            return i['gsd']
    print(f'GSD Missing: Image {im_id}')
    return None

def get_im_info(im_id, gt_content):
    '''
    PURPOSE: Get the info of an image based on its id in a coco file
    IN:
     - im_id: int, image id for the image in question
     - gt_content: the content from a coco ground truth file
    OUT:
     - i: either the image's info or None if it isn'available
    '''
    images = gt_content['images']

    for i in images:
        if i['id'] == im_id:
            return i
    return None


def get_im_ids(gt_json):
    '''
    IN: gt coco json file
    OUT: list of all int image ids in that file
    '''
    im_ids = []
    
    # Open file
    with open(gt_json, 'r') as f:
        gt = json.load(f)
        
    images = gt['images']
    
    # Gather image id from every image
    for i in images:
        im_ids.append(i['id'])
    
    # Double check that it is unique
    im_ids = list(set(im_ids))
    
    return im_ids

def get_im_id_from_name(im_name, gt_content):
    images = gt_content['images']

    for i in images:
        if i['file_name'] == im_name:
            return i['id']
    print('Missing Image', im_name)
    return None

def get_im_name_from_id(im_id, gt_content):
    images = gt_content['images']

    for i in images:
        if i['id'] == im_id:
            return i['file_name']

def get_im_list_percentage(percentage, image_dir):
    '''
    IN: 
       - percentage: int, representing the percentage (out of 100) 
                     of the images you would like to keep
       - image_dir: str, image directory
    OUT: 
       - keep_list: list of str, images to keep
    PURPOSE: Use to create a new ground truth file with a % of your data for 
    faster testing of other procedures on a given dataset
    '''
    im_list = os.listdir(image_dir)
    keep_val = int(len(im_list)*(percentage/100.0))
    random.shuffle(im_list)
    keep_list = im_list[:keep_val]

    return keep_list

def get_obj_size_from_id(cat_id, gt_content):
    '''
    PURPOSE: Get the average size of an object from a coco file, after 
             estimate_category_size has been run on that annotation set
    IN:
     - cat_id: the integer id of a coco category
     - gt_content: the content from a coco ground truth file
    OUT:
     - pt: Either the object's average size (float) or None if it isn't recorded
    '''
    cats = gt_content['categories']

    for c in cats:
        if c['id'] == cat_id:
            return c['average_size']
    return None

def gsd_norm(target_gsd, image_dir, ann_path, new_exp_dir):
    # Create new experimental directory
    if not os.path.exists(new_exp_dir):
      os.mkdir(new_exp_dir)

    new_im_dir = new_exp_dir + 'images/'
    new_json = new_exp_dir + ann_path.split('_')[-1]

    if not os.path.exists(new_im_dir):
      os.mkdir(new_im_dir)

    with open(ann_path, 'r') as f:
      gt = json.load(f)
    
    new_gt = gt.copy()
    new_gt['images'] = []
    new_gt['annotations'] = []

    im_ids = get_im_ids(ann_path)

    for im_id in tqdm(im_ids):

        im_info = get_im_info(im_id, gt)
        
        
        new_im_info = im_info.copy()

        #pull old image information
        old_im_h = im_info['height']
        old_im_w = im_info['width']
        # Get old gsd
        old_gsd = im_info['gsd']
        
        anns = anns_on_image(im_id, gt)
        
        try:
            if old_gsd is not None:

                # Create new image height and width
                new_im_h = int(float((old_im_h*old_gsd)/target_gsd))
                new_im_w = int(float((old_im_w*old_gsd)/target_gsd))

                new_im_info['height'] = new_im_h
                new_im_info['width'] = new_im_w
                new_im_info['gsd'] = target_gsd

                new_gt['images'].append(new_im_info)
                
                new_path = new_im_dir + im_info['file_name']
                old_path = image_dir + im_info['file_name']
                
                img = Image.open(old_path)
                
                img = img.resize((new_im_w, new_im_h))
                
                img.save(new_path)

                for a in anns:
                    new_a = a.copy()

                    # get current annotation bbox for each annotation
                    [old_x1, old_y1, old_w, old_h] = a['bbox']

                    # Created scaled versions of the bbox
                    scaled_x1 = old_x1/old_im_w
                    scaled_w = old_w/old_im_w

                    scaled_y1 = old_y1/old_im_h
                    scaled_h = old_h/old_im_h

                    # Create new bbox
                    x1 = scaled_x1 * new_im_w
                    w = scaled_w * new_im_w

                    y1 = scaled_y1 * new_im_h
                    h = scaled_h * new_im_h

                    new_a['bbox'] = [x1, y1, w, h]

                    new_gt['annotations'].append(new_a)

                
        except:
            print(im_id, 'problem')
            

    with open(new_json, 'w') as f:
        json.dump(new_gt, f)

    return


def gt_from_im_folder(full_gt, img_folder, new_gt_path):
    # Read in full gt
    with open(full_gt, 'r') as f:
        gt = json.load(f)

    # Initialize key storage containers
    contents = gt.copy()
    contents['images'] = []
    contents['annotations'] = []
    images = []
    annotations = []

    # Process one image at a time
    ims = os.listdir(img_folder)

    for image in tqdm(ims, desc = 'building annotations file'):
        i = int(image.split('_')[0])
        anns = anns_on_image(i, gt) 
        annotations.extend(anns)
        for im_info in gt['images']:
            if im_info['id'] == i:
                images.append(im_info)

    # Load new data into appropriate format and save
    contents['images'] = images
    contents['annotations'] = annotations

    if os.path.exists(new_gt_path):
        os.remove(new_gt_path)

    with open(new_gt_path, 'w') as f:
        json.dump(contents, f)

    return

def gt_from_im_list(full_gt, img_list, new_gt_path):
    # Read in full gt
    with open(full_gt, 'r') as f:
        gt = json.load(f)

    # Initialize key storage containers
    contents = gt.copy()
    contents['images'] = []
    contents['annotations'] = []
    images = []
    annotations = []

    # Process one image at a time
    for image in tqdm(img_list):
        i = get_im_id_from_name(image, gt)
        anns = anns_on_image(i, gt) 
        annotations.extend(anns)
        for im_info in gt['images']:
            if im_info['id'] == i:
                images.append(im_info)

    # Load new data into appropriate format and save
    contents['images'] = images
    contents['annotations'] = annotations

    if os.path.exists(new_gt_path):
        os.remove(new_gt_path)

    with open(new_gt_path, 'w') as f:
        json.dump(contents, f)

    return

def json_fewer_cats(old_json, cat_list, ims_no_anns = False, renumber_cats = True):
    '''
    PURPOSE: Create a json with a subset of object categories
    IN:
        -old_json: gt coco json
        -cat_list: list of int category ids to be included in new coco gt json
        -ims_no_anns: if False (default), remove images without annotations from 'images', else keep all original images
    OUT: (new_name) path to new json file 
    '''
    # Name new json by the number of categories being included
    new_name = old_json.replace('.', '_{}.'.format(len(cat_list)))
    
    # Open original gt json
    with open(old_json, 'r') as f:
        contents = json.load(f)
        
    # Pull out key sections of old gt 
    annotations = contents['annotations']
    images = contents['images']
    cats = contents['categories']
    
    # Create new json, with blank annotations
    new_json = contents.copy()
    new_json['annotations'] = []
    
    # Feedback
    print(len(annotations), 'annotations found')
    
    # Process annotations, keeping only those which are in the desired categories
    count = 0
    for a in annotations:
        cat_id = a['category_id']
        if cat_id in cat_list:
            new_json['annotations'].append(a)
        count += 1
    
    # If desired, only keep images that have annotations on them
    if not ims_no_anns:
        new_json['images'] = []
        print(len(images), "images in original data")
        # Check each image for annotations
        for i in images:
            anns = []
            im_id = i['id']
            for a in new_json['annotations']:
                if a['image_id'] == im_id:
                    anns.append(a)
            if len(anns) > 1:
                new_json['images'].append(i)
        print(len(new_json['images']), "images in new data")
    
    # If desired, make sure that categories are sequentially numbered
    if renumber_cats:
        final_annotations = []
        cat_id = 1
        new_cats = []
        for cat in cats:
            old_id = cat['id']
            if old_id in cat_list:
                new_cat = cat.copy()
                
                for a in new_json['annotations']:
                    if a['category_id'] == old_id:
                        new_ann = a.copy()
                        new_ann['category_id'] = cat_id
                        final_annotations.append(new_ann)
                
                new_cat['id'] = cat_id
                cat_id += 1
                new_cats.append(new_cat)
        new_json['annotations'] = final_annotations
        
    new_json['categories'] = new_cats
    
    
    # Feedback
    print(len(new_json['annotations']), 'annotations in new file at', new_name)
    
    # Ensure no .json funny business will happen
    if os.path.exists(new_name):
        os.remove(new_name)
    
    # Save new file
    with open(new_name, 'w') as f:
        json.dump(new_json, f)
    
    return new_name

def json_images(gt_json_path):
    '''
    Purpose: return a list of image paths in a specified json
    '''
    with open(gt_json_path, 'r') as f:
        gt = json.load(f)
    
    images = gt['images']
    im_paths = []
    for i in images:
        im_paths.append(i['file_name'])

    return im_paths

def load_test_train_split(test_gt, image_folder):
    test_ims = json_images(test_gt)

    test_folder = image_folder.replace('train', 'test')

    if not os.path.exists(test_folder):
        os.mkdir(test_folder)

    for i in test_ims:
        dst = test_folder + i
        src = image_folder + i

        shutil.move(src, dst)

    return

def make_exp_by_percentage(data_tag, keep_percent, ims_list, anns_list):
    '''
    IN:
      - data_tag: str, describes the dataset
      - keep_percent: int, percentage of each set of images to keep
      - ims_list: list of str, paths to images (same order as anns_list)
      - anns_list: list of str, paths to annotation files (same order as ims_list)
    OUT: None
    '''

    # make new experimental directory
    new_exp_dir = f'{data_tag}_mini_{keep_percent}'

    if not os.path.exists(new_exp_dir):
        os.mkdir(new_exp_dir)

    for i, anns in enumerate(anns_list):
        ### Images ###
        # get a list of images to keep
        ims = os.listdir(ims_list[i])
        random.shuffle(ims)
        keep_ims = int(len(ims)*(float(keep_percent/100.0)))
        new_ims = ims[:keep_ims]

        # create a new image directory
        new_image_dir = f'{new_exp_dir}/{ims_list[i]}'
        if not os.path.exists(new_image_dir):
            os.mkdir(new_image_dir)
        # move the images
        for im in tqdm(new_ims, desc = f'moving images'):
            src = ims_list[i] + im
            dst = new_image_dir + im
            shutil.copy2(src, dst)

        ### Annotations ###
        new_ann_path = f'{new_exp_dir}/{anns}'
        gt_from_im_folder(anns, new_image_dir, new_ann_path)

def make_cat_ids_match(src_anns, match_anns):
    '''
    IN: 
      - src_anns: str, path to the annotations whose category ids will provide
                  the mapping
      - match_anns: str, path to annotations whose categories will be remapped
    OUT: None, the categories will be remapped in place
    PURPOSE: Given two sets of coco annotations whose categories match, 
    ensure that the ids of each category are the same by forcing 
    match_anns categories to match src_anns categories
    '''
    # Open the annotations
    with open(src_anns, 'r') as f:
        src_gt = json.load(f)
    with open(match_anns, 'r') as f:
        match_gt = json.load(f)
    
    # Get the lists of categories
    src_cats = src_gt['categories']
    match_cats = match_gt['categories']

    # Create a mapping from one set of ids to the other
    cat_map = {}
    for c in match_cats:
        cat_map[c['id']] = get_category_id_from_name(c['name'], src_gt)

    # Remap the annotations in match_anns
    new_annotations = []
    for a in match_gt['annotations']:
        new_a = a.copy()
        new_a['category_id'] = cat_map[a['category_id']]
        new_annotations.append(new_a)
    
    match_gt['annotations'] = new_annotations
    match_gt['categories'] = src_cats
    
    # Save out a new file
    os.remove(match_anns)
    with open(match_anns, 'w') as f:
        json.dump(match_gt, f)

    return 

def make_palette(contents):
    categories = contents['categories']
    
    palette = sns.hls_palette(len(categories))
        
    return palette

def map_to_supercategories(anns, new_fp):
    '''
    PURPOSE: Given some inout coco json, map all the annotations to their 
    supercategories for a more generalized experiment
    IN:
      - ann: str, fp to coco annotation file
      - new_fp: str, fp to new coco annotation file
    
    OUT: Nothing, the new file will be created at the specified path
    '''

    # open the annotations
    with open(anns, 'r') as f:
        gt = json.load(f)
    
    ### Categories ###
    old_cats = gt['categories']

    # create the new set of categories
    new_cs = []
    new_cats = []

    for c in old_cats:
      new_c = c['supercategory']
      new_cs.append(new_c)
    # sort the list of possible categories so they always end up in a predictable order
    new_cs = sorted(list(set(new_cs)))

    for i, c in enumerate(new_cs):
      cat = {'id': i + 1, 'name': c, 'supercategory': 'None'}
      new_cats.append(cat)

    # ### Annotations ###
    old_anns = gt['annotations']
    new_anns = []


    for a in tqdm(old_anns):
        new_ann = a.copy()
        a_cat = a['category_id']
        for c in old_cats:
            if c['id'] == a_cat:
                a_cat_name = c['supercategory']
                for nc in new_cats:
                    if nc['name'] == a_cat_name:
                        a_cat_id = nc['id']
                        new_ann['category_id'] = a_cat_id
                        new_anns.append(new_ann)

    ### New File ###
    # Create new json, with blank annotations
    new_json = gt.copy()
    new_json['annotations'] = new_anns
    new_json['categories'] = new_cats

    # ensure the fp does not already exist
    if os.path.exists(new_fp):
        os.remove(new_fp)

    # write out new anns
    with open(new_fp, 'w') as f:
        json.dump(new_json, f)
    
    return 


def plot_model_performance(model_path, title = ''):

    model_info = torch.load(model_path)

    val_losses = model_info['losses val']
    train_losses = model_info['losses train']
    val_stats = model_info['val stats']

    map = []
    x = []
    precision = []
    recall = []

    for k, v in val_stats.items():
        x.append(k)
        map.append(v[0]['mAP'])
        precision.append(v[0]['precision'])
        recall.append(v[0]['recall'])

    map = [a for y, a in sorted(zip(x, map))]
    precision = [a for y, a in sorted(zip(x, precision))]
    recall = [a for y, a in sorted(zip(x, recall))]
    x = sorted(x)


    plt.plot(range(0, len(val_losses)), val_losses, label = 'Validation Loss')
    plt.plot(range(0, len(train_losses)), train_losses, label = 'Train Loss')
    plt.plot(x, map, label = 'Mean Average Precision')
    plt.plot(x, recall, label = 'Recall')
    plt.plot(x, precision, label = 'Precision')
    plt.legend(loc="upper right")
    plt.title(title)

    return

def random_shift_point(pt, max_shift = 5):
    '''
    PURPOSE: Shift a point at random
    IN:
     - pt: [x,y]
     - max_shift: int, maximum distance the point may shift 
    OUT:
     - pt: [x,y]
    '''
    # unpack the point
    x,y = pt

    # define options
    vert_opts = ['up', 'down', 'centered']
    hori_opts = ['left', 'right', 'centered']

    # randomly select a movement
    v_d = random.choice(vert_opts)
    h_d = random.choice(hori_opts)

    # select shift amount
    v_s = random.choice(range(1, max_shift+1))
    h_s = random.choice(range(1, max_shift+1))

    # move the point:
    if v_d == 'up':
        y += v_s
    elif v_d == 'down':
        y -= v_s

    if h_d == 'right':
        x += v_s
    elif h_d == 'left':
        x -= v_s
  
    # make sure there are no negatives
    if x < 0:
       x = 0
    if y < 0:
       y = 0

    return [x,y]

def show_chip_anns(img, anns, gt_path):
    '''
    IN:
        -img: pixels of image chip to be displayed
        -anns: annotations relative to this image
    OUT: display of that chip and annotations
    '''
    with open(gt_path, 'r') as f:
        gt_content = json.load(f)

    # Get Color palette
    pal = make_palette(gt_content)
    
    # Show image on figure
    plt.figure()
    f,ax = plt.subplots(1, figsize = (10,10))
    plt.imshow(img)
    
    # Add annotations
    for a in anns:
        b = a['bbox']
        cat = a['category_id']
        cat_name = get_category_gt(cat, gt_content)
        rect = patches.Rectangle((b[0], b[1]), b[2], b[3], edgecolor = pal[cat-1], facecolor = "none")
        plt.text(b[0], b[1], cat_name, ha = "left", color = 'w')
        ax.add_patch(rect)
    plt.show()
    return

def show_im_chips(im_id, gt, size):
    '''
    IN:
      - gt: opened coco gt
    '''

    im_anns = anns_on_image(im_id, gt)

    im_name = images + str(im_id) + '.tif'
    img = plt.imread(im_name)

    (y,x,c) = img.shape

    num_x = int(x/size)
    num_y = int(y/size)

    chip_num = 0

    for x_i in range(num_x):
        for y_i in range(num_y):
            c_x1 = x_i * size
            c_y1 = y_i * size
            bbox = [c_y1, c_x1, size, size]
            anns = get_anns_in_box(bbox, im_anns)
            if len(anns) > 0:
                chip_name = str(chip_num) + '_' + str(im_id) + '_{}_{}_{}_{}'.format(c_x1, c_y1, size, size) + '.tif'
                chip_num += 1
                image_chip = img[c_x1:c_x1 + size, c_y1:c_y1 + size]
                show_chip_anns(image_chip, anns, gt)
    
    return

def subchip_images(gt, image_folder, new_image_folder, chip_size):
    '''
    Purpose: Take a coco style json and associated image folder, 
    and create a new coco json and image folder containing new images 
    of the specified size
    '''
    
    # Open gt json
    with open(gt, 'r') as f:
        gt_og = json.load(f)
    
    # New data
    new_images = []
    new_anns = []
    
    # Create new save locations
    gt_new_path = gt.replace('.', '_{}.'.format(chip_size))
   
    if not os.path.exists(new_image_folder):
        os.mkdir(new_image_folder)
    
    chip_num = 0
    
    # Iterate through data one image at a time
    image_ids = get_im_ids(gt)
    images_processed = 0
    for im_id in tqdm(image_ids):
        # Get all original annotations on this image
        im_anns = anns_on_image(im_id, gt_og)
        
        # Open the image
        im_str = get_im_name_from_id(im_id, gt_og)
        im_name = image_folder + im_str
        if os.path.exists(im_name):
            img = plt.imread(im_name)
            
            # Get image dimensions
            try:
              (x,y,c) = img.shape
            except:
              (x,y) = img.shape
            num_x = int(x/chip_size)
            num_y = int(y/chip_size)
            
            # Process each individual chip on this image
            for x_i in range(num_x):
                for y_i in range(num_y):
                    
                    # Get chip coords
                    c_x1 = x_i * chip_size
                    c_y1 = y_i * chip_size
                    bbox = [c_y1, c_x1, chip_size, chip_size]
                    
                    # If there are annotations on this image, save out a chip
                    anns = get_anns_in_box(bbox, im_anns)
                    if len(anns) > 0:
                        chip_name = str(chip_num) + '_' + str(im_id) + '_{}_{}_{}_{}'.format(c_x1, c_y1, chip_size, chip_size) + '.png'
                        chip_path = new_image_folder + chip_name
                        
                        # Update image annotation
                        new_image = {
                            'file_name' : chip_name,
                            'width' : chip_size,
                            'height' : chip_size,
                            'id' : chip_num,
                            'license' : 1
                        }
                        new_images.append(new_image)
                        
                        # Update object annotations
                        for a in anns:
                            new_a = a.copy()
                            new_a['image_id'] = chip_num
                            new_anns.append(new_a)
                        
                        
                        chip_num += 1
                        image_chip = img[c_x1:c_x1 + chip_size, c_y1:c_y1 + chip_size]
                        
                        if not os.path.exists(chip_path):
                            try:
                                plt.imsave(chip_path, image_chip)
                            except:
                                continue
            
                            
            images_processed += 1
        else:
            print(f'Issue with {im_id}')
        
    # Save out new gt file
    new_gt = gt_og.copy()
    new_gt['images'] = new_images
    new_gt['annotations'] = new_anns
    
    if os.path.exists(gt_new_path):
        os.remove(gt_new_path)
    
    with open(gt_new_path, 'w') as f:
        json.dump(new_gt, f)
        
    print('New ground truth:', gt_new_path)
    print('New images:', new_image_folder)
    
    return

def subchip_images_colab(gt, image_folder, new_image_folder, chip_size):
    '''
    Purpose: Take a coco style json and associated image folder of image folders, 
    and create a new coco json and image folder containing new images 
    of the specified size
    '''
    
    # Open gt json
    with open(gt, 'r') as f:
        gt_og = json.load(f)
    
    # New data
    new_images = []
    new_anns = []
    
    # Create new save locations
    gt_new_path = gt.replace('.', '_{}.'.format(chip_size))
   
    if not os.path.exists(new_image_folder):
        os.mkdir(new_image_folder)
        
    # Get paths to all available images in sub-folders
    image_paths = get_image_paths(image_folder)    
    print(len(image_paths), 'images discovered in folders')
    
    chip_num = 0
    
    # Iterate through data one image at a time
    image_ids = get_im_ids(gt)
    images_processed = 0
    for im_id in tqdm(image_ids):
        # Get all original annotations on this image
        im_anns = anns_on_image(im_id, gt)
        
        # Open the image
        im_name = find_im_path(im_id, image_paths)
        if os.path.exists(im_name):
            img = plt.imread(im_name)
            
            # Get image dimensions
            (x,y,c) = img.shape
            num_x = int(x/chip_size)
            num_y = int(y/chip_size)
            
            # Process each individual chip on this image
            for x_i in range(num_x):
                for y_i in range(num_y):
                    
                    # Get chip coords
                    c_x1 = x_i * chip_size
                    c_y1 = y_i * chip_size
                    bbox = [c_y1, c_x1, chip_size, chip_size]
                    
                    # If there are annotations on this image, save out a chip
                    anns = get_anns_in_box(bbox, im_anns)
                    if len(anns) > 0:
                        chip_name = str(chip_num) + '_' + str(im_id) + '_{}_{}_{}_{}'.format(c_x1, c_y1, chip_size, chip_size) + '.png'
                        chip_path = new_image_folder + chip_name
                        
                        # Update image annotation
                        new_image = {
                            'file_name' : chip_name,
                            'width' : chip_size,
                            'height' : chip_size,
                            'id' : chip_num,
                            'license' : 1
                        }
                        new_images.append(new_image)
                        
                        # Update object annotations
                        for a in anns:
                            new_a = a.copy()
                            new_a['image_id'] = chip_num
                            new_anns.append(new_a)
                        
                        
                        chip_num += 1
                        image_chip = img[c_x1:c_x1 + chip_size, c_y1:c_y1 + chip_size]
                        image_chip = Image.fromarray(image_chip).convert('RGB')
                        
                        if not os.path.exists(chip_path):
                            try:
                                image_chip.save(chip_path)
                            except:
                                continue
                            
            images_processed += 1
        
    # Save out new gt file
    new_gt = gt_og.copy()
    new_gt['images'] = new_images
    new_gt['annotations'] = new_anns
    
    if os.path.exists(gt_new_path):
        os.remove(gt_new_path)
    
    with open(gt_new_path, 'w') as f:
        json.dump(new_gt, f)
        
    print('New ground truth:', gt_new_path)
    print('New images:', new_image_folder)
    
    return gt_new_path

def test_train_split(chip_folder, test_percentage, gt, chip_size):
    
    all_chips = os.listdir(chip_folder)

    random.shuffle(all_chips)
    num_chips = len(all_chips)
    print("Num chips", num_chips)

    test_ind = int(num_chips*test_percentage)
    print("Test_ind", test_ind)

    # Select images for test and train
    test_chips = all_chips[:test_ind]
    train_chips = all_chips[test_ind:]
    
    # Make new gt
    train_data_name = 'train_' + str(chip_size) + '_gt.json'
    test_data_name = 'test_' + str(chip_size) + '_gt.json'
    gt_from_im_list(gt, train_chips, train_data_name)
    gt_from_im_list(gt, test_chips, test_data_name)


    print(train_data_name, test_data_name)
    return train_data_name, test_data_name

def train_val_test_split(image_folder, gt, val_precentage = 0.1, test_percentage = 0.2, data_tag = ""):
    
    data_folder = '/'.join(image_folder.split('/')[:-2]) + '/'

    all_images = os.listdir(image_folder)

    random.shuffle(all_images)
    num_images = len(all_images)
    print("Num images", num_images)

    test_ind = int(num_images*test_percentage)
    val_ind = int(num_images*val_precentage) + test_ind

    # Select images for test and train
    test_images = all_images[:test_ind]
    val_images = all_images[test_ind:val_ind]
    train_images = all_images[val_ind:]
    
    # Make new gt
    train_data_name = data_folder + 'train_' + data_tag + '_gt.json'
    val_data_name = data_folder + 'val_' + data_tag + '_gt.json'
    test_data_name = data_folder + 'test_' + data_tag + '_gt.json'

    # Create new gt files
    gt_from_im_list(gt, train_images, train_data_name)
    gt_from_im_list(gt, val_images, val_data_name)
    gt_from_im_list(gt, test_images, test_data_name)
    print('gt created')
    
    # Create new image folders
    train_folder = data_folder + 'train_images/'
    val_folder = data_folder + 'val_images/'
    test_folder = data_folder + 'test_images/'

    # Move all image files as appropriate
    os.mkdir(val_folder)
    os.mkdir(test_folder)

    for i in val_images:
        src = image_folder + i
        dst = val_folder + i
        shutil.move(src, dst)
    
    for i in test_images:
        src = image_folder + i
        dst = test_folder + i
        shutil.move(src, dst) 

    os.rename(image_folder, train_folder)      

    return data_folder 
