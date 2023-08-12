import segmentation_module
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as transforms
import torch
import argparse
import os
import logging
import json



# parser = argparse.ArgumentParser("Run Dino", add_help=True)
# parser.add_argument("--input", "-i", type=str, required=True, help="Directory of input data")
# parser.add_argument("--params", "-p", type=str, required=True, help="Path to the Dino Params JSON")
# parser.add_argument("--output", "-o", type=str, required=True, help="Directory of output path")
# args = parser.parse_args()

DINO_PARAMS = "dino_params.json"
DINO_WEIGHTS = "dino_files/weights/groundingdino_swint_ogc.pth"
DINO_CONFIG = "dino_files/config/GroundingDINO_SwinT_OGC.py"

SAM_WEIGHTS = "model_checkpoint/sam_vit_h_4b8939.pth"
SAM_PARAMS = "sam_params.json"

DATASET = "data/test_images_selection_orig"
DOWNSAMPLED_DATA = "data/test_images_selection"
OUTDIR = "output/results"

# run dino 
dino = segmentation_module.Dino(DINO_WEIGHTS, config= DINO_CONFIG, params=DINO_PARAMS)
sam = segmentation_module.Sam(SAM_WEIGHTS, params=SAM_PARAMS)

# load data
data = segmentation_module.CustomData(DATASET, data_model="dino", resize=800)
data_files = data.data_items
allboxes = []
for ind1, im in enumerate(data):
    # create a downscaled copy for SCALABLE annotation app
    topil = transforms.ToPILImage()
    pilim = topil(im)
    pilim.save(os.path.join(DOWNSAMPLED_DATA, data_files[ind1]))

    # process boxes
    boxes, _ = dino.infer_bbox(im)
    # crop boxes from image
    dino_boxes = []
    for ind2, box in enumerate(boxes):
        # avoid full page boxes
        if box[2]>0.8 and box[3]>0.8:
            continue

        absbox = segmentation_module.rel2abs(box, (im.shape[2], im.shape[1]))
        dino_boxes.append(absbox)
        
        # save crops
        crop = im[:, absbox[1]:absbox[3], absbox[0]:absbox[2]]*255
        crop = np.array(crop).astype(np.uint8).transpose(1,2,0)
        crop = Image.fromarray(crop, mode="RGB")
        savepath = os.path.join(OUTDIR, '{}_{}_{}.jpg'.format(data_files[ind1], ind1, ind2))
        #crop.save(savepath)
    

    # TODO fix in scalable object - for now this is a dirty workaround
    if len(dino_boxes) == 0:
        logging.warning("Page without objects - dirty workaroud...")
        # append a Zeros bbox to avoid index issues. 
        dino_boxes.append([0,0,0,0])
    # pass values to SAM


    sam.predictor.set_image(im)
    sam_boxes = torch.tensor(dino_boxes, device=sam.predictor.device)
    sam_boxes = sam.predictor.transform.apply_boxes_torch(sam_boxes, im.shape[:2])
    masks, _, _ = sam.predictor.predict_torch(point_coords=None, point_labels=None, boxes = sam_boxes) 
    allboxes.append(dino_boxes)
    # save boxes and masks to Scalable

# create a data object for downsampled iterms
down_data = segmentation_module.CustomData(DOWNSAMPLED_DATA, data_model='dino')
sca_bb = segmentation_module.scalable(down_data, allboxes, region_type='bbox')
sca_boxes = sca_bb.create_scalable_project()
with open("run_dino_results.json", 'w') as f:
    json.dump(sca_boxes, f, indent=4)

    

    




    

        



