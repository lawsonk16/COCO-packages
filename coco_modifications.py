from tqdm import tqdm
import json
import os

def clip_anns_to_image(coco_fp):
    '''
    Parameters
    ----------
    coco_fp : str,
        file path to a set of coco ground truth annotations

    Modifies the file in place to remove any annotations whose centerpoints 
    are off the image, which can be common in overhead imagery datasets for some
    reason
    -------

    '''
    #Open the file
    with open(coco_fp, 'r') as f:
      anns = json.load(f)
    new_anns = []
    
    # check the annotations
    for a in tqdm(anns['annotations']):
        new_ann = a.copy()
        x1,y1,w,h = a['bbox']
        x2 = x1+w
        y2 = y1+h

        im_id = a['image_id']
        for i in anns['images']:
          if i['id'] == im_id:
            im_w = i['width']
            im_h = i['height']
            
        # check coordinates which are too large 
        if (y2 > im_h) or (x2 > im_w):
            xc = (x1+x2)/2
            yc = (y1+y2)/2
            if (x1 < im_h) and (y2 < im_h):
                if (xc < im_h) and (yc < im_h):
                  if (x2 > im_w):
                    w = im_w - x1
                    x2 = x1 + w
                  if (y2 > im_h):
                    h = im_h - y1
                    y2 = y1 + h
        # check coordinates which are too small         
        if (y1 < 0) or (x2 < 0):
            xc = (x1+x2)/2
            yc = (y1+y2)/2
            if (xc > 0) and (yc > 0):
              if y1 < 0:
                y1 = 0
                h = y2 - y1
              if x1 < 0:
                x1 = 0
                w = x2 - x1
        
        # make sure that after all modifications the annotation still has
        # a width and a height
        if (w > 0) and (h > 0):
                    new_ann['bbox'] = [x1,y1,w,h]
                    new_anns.append(new_ann)
    anns['annotations'] = new_anns
    
    # save out the modified annotations
    if os.path.exists(coco_fp):
      os.remove(coco_fp)
    with open(coco_fp, 'w') as f:
      anns = json.dump(anns, f, indent = 3)
      
    return

def add_geococo_centerpoints_xy(anns_fp, output_fp = False):
    '''

    Parameters
    ----------
    anns_fp : str,
        file path to a set of coco ground truth annotations
    output_fp : boolean, optional
        If an output fp is specified, that's where the modified anns will be 
        writen. Else, a version of the fp with -cp added before .json will be 
        used. The default is False.

    Modifies the file using the bbox key to add centerpoint values to each 
    annotation using the format ['object_center'] = [xc, yc]
    -------

    '''
    
    with open(anns_fp, 'r') as f:
      anns = json.load(f)
    annotations = anns['annotations']
    new_anns = []
    for a in annotations:
      new_a = a.copy()
      [x,y,w,h] = a['bbox']
      xc = int(x+(w/2))
      yc = int(y+(h/2))
      new_a['object_center'] = [xc, yc]
      new_anns.append(new_a)
    anns['annotations'] = new_anns
    if output_fp:
        if os.path.exists(output_fp):
            os.remove(output_fp)
        with open(output_fp, 'w') as f:
          json.dump(anns, f, indent = 5)
    else:
        output_fp = anns_fp.replace('.json', '-cp.json')
        if os.path.exists(output_fp):
            os.remove(output_fp)
        with open(output_fp, 'w') as f:
          json.dump(anns, f, indent = 5)
     
    return

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