# IMAGE 
import cv2
from PIL import Image, ImageDraw, ImageFont
from PIL.ImageOps import autocontrast
from shapely.geometry import Polygon
from skimage import measure


# UTILS
import os
import numpy as np
import matplotlib.pyplot as plt
import json
import uuid
from typing import Union,TypeVar

#SAM
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

#DINO
import groundingdino.datasets.transforms as dino_T
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap


# Torch Import
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F



def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.

    Parameters:
    box1 (list): List containing [x1, y1, x2, y2] of the first bounding box.
    box2 (list): List containing [x1, y1, x2, y2] of the second bounding box.

    Returns:
    float: Intersection over Union (IoU) between the two boxes.
    """
    x1_overlap = max(box1[0], box2[0])
    y1_overlap = max(box1[1], box2[1])
    x2_overlap = min(box1[2], box2[2])
    y2_overlap = min(box1[3], box2[3])

    intersection_area = max(0, x2_overlap - x1_overlap + 1) * max(0, y2_overlap - y1_overlap + 1)

    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    union_area = box1_area + box2_area - intersection_area

    iou = intersection_area / union_area
    return iou


def non_max_suppression(bboxes, iou_threshold):
    """
    Apply Non-Maximum Suppression (NMS) to a list of bounding boxes.

    Parameters:
    bboxes (list): List of bounding boxes in the format (x1, y1, x2, y2, confidence_score).
    iou_threshold (float): IoU threshold for considering bounding boxes as duplicates.

    Returns:
    list: List of selected bounding boxes after NMS.
    """
    # Sort the bounding boxes by confidence score in descending order
    sorted_bboxes = sorted(bboxes, key=lambda x: x[4], reverse=True)

    selected_bboxes = []

    while sorted_bboxes:
        current_bbox = sorted_bboxes.pop(0)
        selected_bboxes.append(current_bbox)

        remaining_bboxes = []
        for bbox in sorted_bboxes:
            iou = calculate_iou(current_bbox[:4], bbox[:4])
            if iou < iou_threshold:
                remaining_bboxes.append(bbox)

        sorted_bboxes = remaining_bboxes

    return selected_bboxes


## show annotations -> from SAM repo
def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    polygons = []
    color = []
    for ann in sorted_anns:
        m = ann['segmentation']#
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:,:,i] = color_mask[i]
        ax.imshow(np.dstack((img, m*0.35)))


## show boxes from DINO output
def plot_boxes_to_image(image_pil, tgt):
    H, W = tgt["size"]
    boxes = tgt["boxes"]
    labels = tgt["labels"]
    assert len(boxes) == len(labels), "boxes and labels must have same length"

    draw = ImageDraw.Draw(image_pil)
    mask = Image.new("L", image_pil.size, 0)
    mask_draw = ImageDraw.Draw(mask)

    # draw boxes and masks
    for box, label in zip(boxes, labels):
        # from 0..1 to 0..W, 0..H
        box = box * torch.Tensor([W, H, W, H])
        # from xywh to xyxy
        box[:2] -= box[2:] / 2
        box[2:] += box[:2]
        # random color
        color = tuple(np.random.randint(0, 255, size=3).tolist())
        # draw
        x0, y0, x1, y1 = box
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

        draw.rectangle([x0, y0, x1, y1], outline=color, width=6)
        # draw.text((x0, y0), str(label), fill=color)

        font = ImageFont.load_default()
        if hasattr(font, "getbbox"):
            bbox = draw.textbbox((x0, y0), str(label), font)
        else:
            w, h = draw.textsize(str(label), font)
            bbox = (x0, y0, w + x0, y0 + h)
        # bbox = draw.textbbox((x0, y0), str(label))
        draw.rectangle(bbox, fill=color)
        draw.text((x0, y0), str(label), fill="white")

        mask_draw.rectangle([x0, y0, x1, y1], fill=255, width=6)

    return image_pil, mask


def rel2abs(box, size):
	# x1 = box[0]*size[0]
	# y1 = box[1]*size[1]
	# x2 = x1 + box[2]*size[0]
	# y2 = y1 + box[3]*size[1]
	# return [x1, y1, x2, y2]
	w, h = size
	newbox = box * np.array([w, h, w, h])

	newbox[:2] -= newbox[2:] / 2
	newbox[2:] += newbox[:2]
	newbox = [int(x) for x in newbox]
	return newbox


class ToTensorNoNorm(transforms.ToTensor):
    """Custom toTensor without scaling"""
    def __call__(self, pic):
        """
        Args:
            pic (PIL Image): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        if type(pic) is not Image.Image:
             raise TypeError('pic should be PIL Image. Got {}'.format(type(pic)))

        # Convert PIL image to tensor without scaling
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        img = img.view(pic.size[1], pic.size[0], len(pic.getbands())).transpose(0, 1).transpose(0, 2).contiguous()
        return img

    
    
class CustomData(Dataset):
    """
    Standard CustomData class with some standard transforms that should be used on all data passed to SAM.
    """

    def __init__(self, data_dir:str, data_model = str, resize=800):
        self.data_root = data_dir
        self.data_items = np.array(os.listdir(data_dir))
        self.resize = resize    # tuple
        if data_model not in {"sam", "dino"}:
            raise TypeError("data_model should specify whether the model is sam or dino, other value give. exiting")
        self.data_model = data_model

    def __len__(self):
        return len(self.data_items)
    
    def __getitem__(self, idx):
        """
        get item to pass to model 
        """

        if torch.is_tensor(idx):
            idx = idx.tolist()

        # open image
        impath = os.path.join(self.data_root, self.data_items[idx])
        # convert to RGB to remove possible Alpha layer
        image = Image.open(impath).convert('RGB')
        

        # transforms are including in the image and are not optional to be given. 
        # only the shape can be as input
        if self.data_model == "sam":
            if self.resize:
                transform = transforms.Compose([
                    transforms.Resize(self.resize), # Resize the image to 
                    ToTensorNoNorm() # Convert the image to a PyTorch tensor without scaling
                ])
            # TODO remove if else condition as size now defaults to 800
            else:
                transform = transforms.Compose([
                    ToTensorNoNorm() # Convert the image to a PyTorch tensor without scaling
                ])
                
            image = transform(image)
            image = image.permute(1,2,0).to(torch.uint8)

            return image
        elif self.data_model == "dino":
            transform = dino_T.Compose(
            [   
                #transforms.Resize(self.resize),
                dino_T.RandomResize([self.resize], max_size=1333),
                dino_T.ToTensor(),
                #dino_T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
            )



            image, _ = transform(image, None)  # 3, h, w

        return image

class Dino:
    """class to initialize Grounding Dino and pass it data"""
    def __init__(self, weights:str, config:str, params:str) -> None:
        self.weights = weights
        self.config = config
        self.model = self.load_model()
        
        with open(params, "r") as f:
            hyperparams = json.load(f)
        self.params = hyperparams

    def load_model(self):
        args = SLConfig.fromfile(self.config)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # TODO MPS still lacks some features. implement this at a later stage. 
        #device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        args.device = device
        model = build_model(args)
        checkpoint = torch.load(self.weights, map_location="cpu")
        load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        print(load_res)
        _ = model.eval()
        return model


    def infer_bbox(self, image:Union[Dataset, torch.Tensor]):
        with_logits = True
        #TODO add options for Dataset and image
        caption_list = self.params["prompt"]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = self.model.to(device)
        image = image.to(device)
        pred_phrases, box_pred, box_only = [], [], []
        for caption in caption_list:
            caption = caption.lower().strip()
            if not caption.endswith("."):
                caption = caption + "."
            #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            #device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
            # model = self.model.to(device)
            # image = image.to(device)

            with torch.no_grad():
                outputs = model(image[None], captions=[caption])
            logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
            boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
            logits.shape[0]
            logits_filt = logits.clone()
            boxes_filt = boxes.clone()
            filt_mask = logits_filt.max(dim=1)[0] > self.params["box_threshold"]
            logits_filt = logits_filt[filt_mask]  # num_filt, 256
            boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
            logits_filt.shape[0]

            tokenlizer = model.tokenizer
            tokenized = tokenlizer(caption)
            # build pred
            # pred_phrases, box_pred = [], []
            for logit, box in zip(logits_filt, boxes_filt):

                #TODO added confidence scores here
                var = list(torch.cat((box, torch.Tensor(logit.max()).unsqueeze(0))))
                box_pred.append(var)
                box_only.append(box)
                pred_phrase = get_phrases_from_posmap(logit > self.params["text_threshold"], tokenized, tokenlizer)
                if with_logits:
                    pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
                else:
                    pred_phrases.append(pred_phrase)
        return box_only, pred_phrases, torch.Tensor(box_pred)



class Sam:
    # box format xyxy
    """
    Class to package sam load and run into short commands
    """
    DEFAULT_PARAMS = """
    {
        "points_per_side":32,
        "points_per_batch":64,
        "pred_iou_thresh":0.88,
        "stability_score_thresh":0.95,
        "stability_score_offset":1,
        "box_nms_thresh":0.7,
        "crop_n_layers":0,
        "crop_nms_thresh":0.7, 
        "crop_overlap_ratio": 0.3, 
        "crop_n_points_downscale_factor":1,
        "min_mask_region_area":0
    }
    """

    def __init__(self, checkpoint:str, params=None) -> None:
        self.checkpoint = checkpoint
        self.model_type = 'vit_h' 
        
        if params:
            with open(params, 'r') as f:
                hyparams = json.load(f)
            self.params = hyparams
        else:
            self.params = json.loads(Sam.DEFAULT_PARAMS)

        self.mask_generator, self.predictor = self.load_model()

    def load_model(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sam = sam_model_registry[self.model_type](checkpoint=self.checkpoint) 
        sam.to(device=device) 

        mask_gen = SamAutomaticMaskGenerator(
            model = sam, 
            points_per_side=self.params['points_per_side'], 
            points_per_batch=self.params['points_per_batch'],
            pred_iou_thresh=self.params['pred_iou_thresh'], 
            stability_score_thresh=self.params['stability_score_thresh'], 
            stability_score_offset=self.params['stability_score_offset'], 
            box_nms_thresh=self.params['box_nms_thresh'], 
            crop_n_layers=self.params['crop_n_layers'], 
            crop_nms_thresh=self.params['crop_nms_thresh'], 
            crop_overlap_ratio=self.params['crop_overlap_ratio'], 
            crop_n_points_downscale_factor=self.params['crop_n_points_downscale_factor'], 
            min_mask_region_area=self.params['min_mask_region_area'])
        predictor = SamPredictor(sam)
        
        return mask_gen, predictor

class scalable():
    """
    Class to process the masks generated from SAM into scalable.ai compatible files
    """
    DEFAULT_CLASSES = {
        "class_list":["0_default"]
    }

    def __init__(self, data:CustomData, masks:list, classes=None, simplification_ratio = 10, region_type = 'mask') -> None:
        accepted_types = {'bbox', 'mask'}

        if region_type.lower() not in accepted_types:
            raise ValueError("accepted_types should be either 'bbox' or 'mask' -> default value is mask")

        self.data = data
        if classes:
            with open(classes, 'r') as f:
                var = json.load(f)
            self.classes = var['class_list']
        else:
            self.classes = scalable.DEFAULT_CLASSES["class_list"]
        self.mask = masks
        self.simplification_ratio = simplification_ratio
        self.region_type = region_type

    def create_scalable_project(self):
        data = self.data
        app_prefix = os.path.join('items', data.data_root)
        item_list = []
        for ind, imname in enumerate(data.data_items):
            imname = os.path.join(app_prefix, imname)
            item = self.construct_item(imname, self.mask[ind])
            item_list.append(item)
        
        classes_list = self.construct_classes()

        scalable_object = {}
        scalable_object['frames'] = item_list
        scalable_object['config'] = classes_list
        return scalable_object


    def construct_classes(self):
        classes_list = []
        for cls in self.classes:
            obj = {}
            obj['name'] = cls
            classes_list.append(obj)
        cls_obj = {}
        cls_obj['attributes'] = []
        cls_obj['categories'] = classes_list
        return cls_obj
        

       
    def construct_item(self, name, mask):
        region_type = self.region_type
        label_list = []
        for m in mask:
            label_object = {}
            label_object['id'] = str(uuid.uuid1())
            # always add default class 
            label_object['category'] = scalable.DEFAULT_CLASSES['class_list'][0]
            
            if region_type.lower() == 'mask':
                #TODO here is the issue, m[segmentation] is part of the general output. should be adjusted for a specific prompt.
                if type(m) == dict:
                    current_mask = m["segmentation"]
                else:
                    current_mask = m[0]
                
                geom_coords = self.mask2poly(current_mask)
                # turn geom_coords into a list of lists to match scalable schema
                geom_coords = [list(sublist) for sublist in geom_coords]
                geom_type = "".join(['L' for i in range(len(geom_coords))])
                poly_object = {}
                poly_object['vertices'] = geom_coords
                poly_object['types'] = geom_type
                poly_object['closed'] = True
                label_object['poly2d'] = [poly_object]

            elif region_type.lower() == 'bbox':
                if type(m) == dict:
                    current_box = m["bbox"]
                else:
                    current_box = m
                geom_coords = current_box
                box_object = {}
                # TODO check if reverse order is needed
                box_object['x1'] = geom_coords[0]
                box_object['y1'] = geom_coords[1]
                box_object['x2'] = geom_coords[2]
                box_object['y2'] = geom_coords[3]
                # TODO CHECK BOXES and DIMS
                # box_object['x2'] = geom_coords[0] + geom_coords[2]
                # box_object['y2'] = geom_coords[1] + geom_coords[3]
                label_object['box2d'] = box_object

            label_list.append(label_object)
    
        item = {}
        item['url'] = name
        item['name'] = name
        item['videoName'] = 'var'
        item['labels'] = label_list
        item['sensor'] = -1
        return item


    def mask2poly(self, mask):
        sim_ratio = self.simplification_ratio
        marray = mask
        #marray = mask['segmentation']
        contours = measure.find_contours(marray, 0.5)
        # Convert the contour to a polygon
        polygon = contours[0].tolist()
        polygon = [tuple(sublist) for sublist in polygon]
        polygon = Polygon(polygon)
        polygon_simple = polygon.simplify(sim_ratio, preserve_topology=True)
        coords = list(polygon_simple.exterior.coords)
        coords = coords[:-1]  # remove last item to avoid intersection  - scalabel compliant
        coords = [[sublist[1], sublist[0]] for sublist in coords]   # switch to image coords (y,x)
        return coords