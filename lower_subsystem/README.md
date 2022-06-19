# Lower Subsystem

## Instructions

### Installation
* Python dependencies:
```
pip3 install -r requirements.txt
```

* Baseline: download the [Mask R-CNN baseline](https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x/138205316/model_final_a3ec72.pkl) and place it inside _./baselines/R101-FPN/_

### Source Segmentator
* Train the source segmentator (SSeg)
```
python3 train_net.py --config-file ./configs/sseg.yaml
```

### Target Segmentator
* Generate pseudo-masks
```
python3 generate_pseudomasks.py --config-file ./configs/tseg.yaml
```

* Train target segmentator (TSeg)
```
python3 train_net.py --config-file ./configs/tseg.yaml
```
