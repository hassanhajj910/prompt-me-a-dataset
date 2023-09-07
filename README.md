# promt-me-a-dataset
Repo used to generate an image dataset and evaluate the results using two chained models. A Zero shot object detection [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) and a [Segment Anything Model](https://github.com/facebookresearch/segment-anything) (SAM). The results are then returned as a json that can be easily processed within the [scalabel.ai](https://github.com/scalabel/scalabel) application. 


# Info

In this paper, we present a pipeline for image extraction from historical documents using foundation models, and evaluate the text-image prompts and their effectiveness on humanities datasets of varying difficulties. The motivation of this approach stems from the high interest of historians in visual elements printed alongside historical texts on the one hand, and from the relative lack of well annotated datasets within the humanities compared to other domains. We propose a sequential approach that relies on GroundDINO and Meta's Segment-Anything-Model (SAM) to retrieve a significant portion of visual data from historical documents which can then to be used for downstream development tasks and dataset creation, and evaluate the effect of different linguistic prompts of the resulting detections.

The idea behind this repo is that, especially within the humanities, access to very specific types of annotated data remains very limited. One of the biggest hurdles here is the level of investment that humanities institutes put into creating such datasets (with the absence of an industry incentive). By using and adapting two large models and harnessing their Zero shot inference, we can tweak the language prompts to quickly build _base_ datasets from humanities images to be later used to train more task specific models. This would circumnavigate one of the hurdles in humanities dataset generation, and speed up this process. 

# How to use

install both groundingDINO and SAM:
```
pip install git+https://github.com/facebookresearch/segment-anything.git
pip install git+https://github.com/IDEA-Research/GroundingDINO.git
```

Download the SAM weights and save them in: 
```
sam_files/weights/
```

Download the GroundingDINO files and save them in:
```
dino_files/weights/
```

language prompts and dino main params are set in the 
```
dino_params.json
```

model params for sam are set in 
```
sam_params.json
```

To run the pipeline set the above and run:
```
python run_pipeline.py -i data/dataset/
```

The results will be saved as a json that can be directly imported into the scalabel application for correction or manipulated for any downstream task.

## To cite this work:

```
@misc{elhajj2023prompt,
      title={Prompt me a Dataset: An investigation of text-image prompting for historical image dataset creation using foundation models}, 
      author={Hassan El-Hajj and Matteo Valleriani},
      year={2023},
      eprint={2309.01674},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
```


## The work relies on the below research:
```
@article{liu2023grounding,
  title={Grounding dino: Marrying dino with grounded pre-training for open-set object detection},
  author={Liu, Shilong and Zeng, Zhaoyang and Ren, Tianhe and Li, Feng and Zhang, Hao and Yang, Jie and Li, Chunyuan and Yang, Jianwei and Su, Hang and Zhu, Jun and others},
  journal={arXiv preprint arXiv:2303.05499},
  year={2023}
}
```

```
@article{kirillov2023segany,
  title={Segment Anything},
  author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C. and Lo, Wan-Yen and Doll{\'a}r, Piotr and Girshick, Ross},
  journal={arXiv:2304.02643},
  year={2023}
}
```