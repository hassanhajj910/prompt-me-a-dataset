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

SAM_WEIGHTS = "sam_files/weights/sam_vit_h_4b8939.pth"
SAM_PARAMS = "sam_params.json"

DATASET = "data/dataset"
DOWNSAMPLED_DATA = "data/dataset_down"
OUTDIR = "output/results"

# run dino 
dino = segmentation_module.Dino(DINO_WEIGHTS, config= DINO_CONFIG, params=DINO_PARAMS)
sam = segmentation_module.Sam(SAM_WEIGHTS, params=SAM_PARAMS)

# load data
data = segmentation_module.CustomData(DATASET, data_model="dino", resize=800)
data_files = data.data_items
allboxes, allmasks = [], []
for ind1, im in enumerate(data):
    # create a downscaled copy for SCALABLE annotation app
    topil = transforms.ToPILImage()
    pilim = topil(im)
    pilim.save(os.path.join(DOWNSAMPLED_DATA, data_files[ind1]))

    # process boxes
    boxes, _, boxes_p = dino.infer_bbox(im)
    # crop boxes from image
    dino_boxes, dino_boxes_p = [], []
    for ind2, (box, box_p) in enumerate(zip(boxes, boxes_p)):
        # avoid full page boxes
        if box[2]>0.8 and box[3]>0.8:
            continue

        absbox = segmentation_module.rel2abs(box, (im.shape[2], im.shape[1]))
        dino_boxes.append(absbox)
        c = box_p[4].item()
        dino_boxes_p.append(absbox[:4] + [c])
        
        # save crops
        crop = im[:, absbox[1]:absbox[3], absbox[0]:absbox[2]]*255
        crop = np.array(crop).astype(np.uint8).transpose(1,2,0)
        crop = Image.fromarray(crop, mode="RGB")
        savepath = os.path.join(OUTDIR, "{}_{}_{}.jpg".format(data_files[ind1], ind1, ind2))
        #crop.save(savepath)

        
    selected_bboxes = segmentation_module.non_max_suppression(dino_boxes_p, 0.5)
    dino_boxes = [x[:4] for x in selected_bboxes]     # remove c
    

    # TODO fix in scalable object - for now this is a dirty workaround
    if len(dino_boxes) == 0:
        logging.warning("Page without objects - dirty workaroud...")
        # append a Zeros bbox to avoid index issues. 
        dino_boxes.append([0,0,0,0])
    # pass values to SAM

    samim = im.permute(1,2,0)
    samim = np.array(samim)*255
    samim = samim.astype(np.uint8)
    sam.predictor.set_image(samim)
    sam.predictor.set_image(samim)
    sam_boxes = torch.tensor(dino_boxes, device=sam.predictor.device)
    sam_boxes = sam.predictor.transform.apply_boxes_torch(sam_boxes, samim.shape[:2])
    masks, _, _ = sam.predictor.predict_torch(point_coords=None, point_labels=None, boxes = sam_boxes) 

    allboxes.append(dino_boxes)
    allmasks.append(np.array(masks))
    # save boxes and masks to Scalable

# create a data object for downsampled iterms
down_data = segmentation_module.CustomData(DOWNSAMPLED_DATA, data_model="dino")
sca_bb = segmentation_module.scalable(down_data, allboxes, region_type="bbox")
sca_boxes = sca_bb.create_scalable_project()
with open(os.path.join(OUTDIR, "run_dino_results_box.json"), "w") as f:
    json.dump(sca_boxes, f, indent=4)


sca_bb = segmentation_module.scalable(down_data, allmasks, region_type="mask")
sca_boxes = sca_bb.create_scalable_project()
with open(os.path.join(OUTDIR, "run_dino_results_masks.json"), "w") as f:
    json.dump(sca_boxes, f, indent=4)

    

    




    

        



