from matplotlib import pyplot as plt
import json
import seaborn as sns
import random
from matplotlib import patches

'''########################### Helper Functions ########################### '''

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

def make_palette(contents):
    categories = contents['categories']
    
    palette = sns.hls_palette(len(categories))
        
    return palette



'''############################ Ground Truth ############################ '''

def display_random_gt(num_ims, json_path, image_folder, fig_size = (20,20), text_on = True):
    '''
    PURPOSE: Display some number of images and their ground truth labels from a coco dataset, randomly selected
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

def display_specific_gt(im_ids, json_path, image_folder, fig_size = (20,20), text_on = True):
    '''
    PURPOSE: Display some number of images and their ground truth labels from a coco dataset, randomly selected
    IN:
        -im_ids: list of ints indicating the image_ids to be displayed
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

    # Process each image
    for i in im_ids:
        
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


def display_random_gt_cp(num_ims, json_path, image_folder, fig_size = (20,20), text_on = True, radius = 2):
    '''
    PURPOSE: Display some number of images from a coco dataset with 'centerpoint' key, randomly selected
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
        plt.imshow(img)
        for a in anns:
            b = a['centerpoint']
            cat = a['category_id']
            cat_name = get_category_gt(cat, gt)
            rect = patches.Circle((b[0], b[1]), radius = radius, edgecolor = pal[cat-1], facecolor = pal[cat-1])
            if text_on:
                plt.text(b[0], b[1], cat_name, ha = "left", color = 'w')
            ax.add_patch(rect)
        plt.title(im_name)
        plt.show()
    
    return

def display_specific_gt_cp(im_ids, json_path, image_folder, fig_size = (20,20), text_on = True, radius = 2):
    '''
    PURPOSE: Display some number of images from a coco dataset with 'centerpoint' key, specifically selected
    IN:
        -im_ids: list of ints indicating the image_ids to be displayed
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

    # Process each image
    for i in im_ids:
        
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
        plt.imshow(img)
        for a in anns:
            b = a['centerpoint']
            cat = a['category_id']
            cat_name = get_category_gt(cat, gt)
            rect = patches.Circle((b[0], b[1]), radius = radius, edgecolor = pal[cat-1], facecolor = pal[cat-1])
            if text_on:
                plt.text(b[0], b[1], cat_name, ha = "left", color = 'w')
            ax.add_patch(rect)
        plt.title(im_name)
        plt.show()
    
    return

'''############################# Detections ############################# '''

def display_random_dt(num_ims, gt_path, dt_path, image_folder, fig_size = (20,20), conf_thresh = 0.9):
    '''
    PURPOSE: Display some number of images and trheir detections cfrom a coco dataset, randomly selected
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

def display_specific_dt(im_ids, gt_path, dt_path, image_folder, fig_size = (20,20), conf_thresh = 0.9):
    '''
    PURPOSE: Display a specific set of images and their detections from a coco dataset
    IN:
        -im_ids: list of ints indicating the image_ids to be displayed
        -json_path: coco gt file
        -image_folder: folder where images in json_path are located
    OUT:
        -figures with each randomly selected image and its annotations
    '''
    with open(gt_path, 'r') as f:
        gt_content = json.load(f)

    # Get Color palette
    pal = make_palette(gt_content)
    
    with open(gt_path, 'r') as f:
        gt = json.load(f)
        
    # Process each image
    for i in im_ids:
        
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

'''##################### Ground Truth and Detections ##################### '''


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

def display_specific_gt_dt(im_ids, gt_path, dt_path, image_folder, fig_size = (20,20), conf_thresh = 0.9):
    '''
    PURPOSE: Display a specific set of images and their detections from a coco dataset
    IN:
        -im_ids: list of ints indicating the image_ids to be displayed
        -json_path: coco gt file
        -image_folder: folder where images in json_path are located
    OUT:
        -figures with each randomly selected image and its annotations
    '''

    with open(gt_path, 'r') as f:
        gt_content = json.load(f)

    # Get Color palette
    pal = make_palette(gt_content)
    
    with open(gt_path, 'r') as f:
        gt = json.load(f)
        
    # Process each image
    for i in im_ids:
        
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

