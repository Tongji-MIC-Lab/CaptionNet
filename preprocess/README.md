# Preprocess
## COCO
### Data Downloading
Put `captions_val2014.json` and `captions_train2014.json` to `../data/annotations`. And download `train2014.zip` and `val2014.zip`, unzip those images, and merge all the images to `../coco_images`.
### Split Dataset
For karpathy split (5000 for validation and 5000 for testing), run:
```bash
python3 coco_split.py --split_option='karpathy'
```
For coco online split (2000 for validation and 0 for testing), run:
```bash
python3 coco_split.py --split_option='online'
```
### Obtain Image Features
For training with Resnet, images must be resized to the same size:
```bash
python3 resize.py --image_dir='../coco_images'  --output_dir=’../coco_resized’  --image_size=576
```
This command will resize all images to 576×576 and store them to `../coco_resized`.
### N-gram Computing
To compute document frequencies of training captions for CIDEr-D reward computing, run:
```bash
python3 prep_ngram.py --input_json='../data/annotations/karpathy_split_train.json'
```
The document frequency will be stored under `../data`. For coco online split, replace the input_json with `online_split_train.json`
### MSCOCO Official Test Server
If you'd like to submit your results to COCO test server, download `test2014.zip` and put the images to `../coco_test2014`
## Flickr30k
### Data Downloading
Put `results_20130124.token` to `../data/annotations`. Download all the images to `../f30k_images`.
### Split Dataset
Run the following command:
```bash
python3 flickr30k_split.py 
```
### Obtain Image Features
For training with Resnet, images must be resized to the same size:
```bash
python3 resize.py --image_dir='../f30k_images'  --output_dir='../f30k_resized’  --image_size=576
```
