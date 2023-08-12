# promt-me-a-dataset
Repo used to generate an image dataset and evaluate the results using two chained models. A Zero shot object detection [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) and a [Segment Anything Model](https://github.com/facebookresearch/segment-anything) (SAM). The results are then returned as a json that can be easily processed within the (scalabel.ai)[https://github.com/scalabel/scalabel] application. 








The work relies on the below research:
'''
@article{liu2023grounding,
  title={Grounding dino: Marrying dino with grounded pre-training for open-set object detection},
  author={Liu, Shilong and Zeng, Zhaoyang and Ren, Tianhe and Li, Feng and Zhang, Hao and Yang, Jie and Li, Chunyuan and Yang, Jianwei and Su, Hang and Zhu, Jun and others},
  journal={arXiv preprint arXiv:2303.05499},
  year={2023}
}
'''

'''
@article{kirillov2023segany,
  title={Segment Anything},
  author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C. and Lo, Wan-Yen and Doll{\'a}r, Piotr and Girshick, Ross},
  journal={arXiv:2304.02643},
  year={2023}
}
'''

To cite this work:

'''
@inproceedings{ElHajj2023
author = {Hassan El-Hajj and Matteo Valleriani},
title = {Prompt me a Dataset: An investigation of text-image prompting for historical image dataset creation using foundation models},
booktitle = {Proceedings of the ICIAP2023 Udine},
year = {2023}
}

'''