# CaptionNet
CaptionNet Based Image to Language
## Requirement
- Python 3.6
- pytorch >= 1.0.1
- Other python packages: nltk, pycocotools
## Installation
Download COCO evaluation tools for python3 from [here](https://github.com/ruotianluo/coco-caption) and put `pycocoevalcap` to the root directory of this repository. And clone CIDEr-D tools for reward computing during reinforcement learning:
```bash
git clone https://github.com/ruotianluo/cider.git
```
## Data Preprocess
- For data preprocessing code and README, navigate to `./preprocess` directory.
- Afterwards, to build the vocabulary for coco dataset:
```bash
python3 build_vocab.py
```
The vocabulary `vocab.pkl` will be saved to `./data`
- To build the vocabulary for flick30k dataset:
```bash
python3 build_vocab.py --caption_path='data/annotations/flikr30k_train.json' --threshold=3 --vocab_path='data/vocab_flickr30k.pkl' --dataset='flickr30k'
```
## Training
### COCO
- Train with Cross-Entropy Loss using Resnet-152:
```bash
python3 training.py --model_path='models/'  --vocab_path='data/vocab.pkl' --image_dir='coco_resized'  --crop_size=512  --caption_path='data/annotations/karpathy_split_train.json' --num_epochs=10  --num_layers=2 --batch_size=150 --device=9  --lr=5e-4 --save_step=1000  --save_name='CAP' --model_name='CAP' --lr_decay_rate=0.8  --lr_decay=21 --lr_decay_every=5  --truncate=18  --dataset='coco'  --use_multi_gpu=True --other_device 7 8   
```
By running the above command, the CaptionNet model will train on 3 GPUs(device 9,8,7, the main device is 9) for 10 epochs. The model will stored every 1000 iterations and every epochs at `./models` .
- Train with Cross-Entropy Loss fine-tuning Resnet-152:
```bash
python3 training.py --model_path='models/'  --vocab_path='data/vocab.pkl' --image_dir='coco_resized'  --crop_size=512  --caption_path='data/annotations/karpathy_split_train.json' --num_epochs=15  --num_layers=2 --batch_size=60 --device=9  --pre_device=9  --lr=5e-4 --save_step=1000 --save_name='CAP' --model_name='CAP --lr_decay_rate=0.8  --lr_decay=21 --lr_decay_every=5  --truncate=18  --dataset='coco'  --use_multi_gpu=True --other_device 5 6 7 8    --load_pre_model=True  --pre_model='CAP-10.ckpt'  --cnn_fine_tune=True --lr_cnn=1e-5  --fine_tune_start_layer=5  
```
This will load pre-trained model `CAP-10.ckpt` and start to fine-tuning Resnet-152 on 5 GPUs(device 9,5,6,7,8, the main device is 9).
- Train with CIDEr-D optimization (self-critical sequence training):
```bash
python3 training.py --model_path='models/'  --vocab_path='data/vocab.pkl' --image_dir='coco_resized'  --crop_size=512  --caption_path='data/annotations/karpathy_split_train.json' --num_epochs=10  --num_layers=2 --batch_size=150 --device=9  --pre_device=9  --lr=5e-5 --save_step=1000 --save_name='CAP' --model_name='CAP --lr_decay_rate=0.8  --lr_decay=26 --lr_decay_every=5  --truncate=18  --dataset='coco'  --use_multi_gpu=True --other_device 7 8    --load_pre_model=True  --pre_model='CAP-25.ckpt' --SCST='t' --disable_dropout='t' --clip=0.1
```
This will load pre-trained model `CAP-25.ckpt` and perform CIDEr-D optimization for 10 epochs.
- For ablation study `Effect of IFE Sequence Length`, change the `sequential_length` argument when training with the `GLO`(i.e. `IFE`) model.
- For ablation study `Contribution of Different Components`, change the `eliminate_code` argument when training with the `CAP` model. 
```bash
eliminate_code = 0: no ablation study
eliminate_code = 1: disable CaptionNet
eliminate_code = 2: disable IFE
eliminate_code = 3: disable both IFE and CaptionNet
```
### Flickr30k
It's easy to change dataset from coco to flickr30k. Simply replace `dataset` argument from `coco` to `flickr30k`, set `image_dir` to `f30k_resized` , change `caption_path` argument to `data/annotations/flickr30k_train.json`, and change `vocab_path` argument to `data/vocab_flickr30k.pkl`.
## Testing
### COCO
- Test on a single image:
```bash
python3 test.py --vocab_path='data/vocab.pkl' --model_path='models/CAP-35.ckpt'  --image_path='sample_picture.jpg'   --eval=0 --beam_size=3  --device=0  --model_device=9  --model_name='CAP' --image_size=512 
```
The `model_device` argument specifies the main device the model is trained, while the `device` argument specifies the device for testing.
- Test on a test split, for instance, `karpathy_split_test.json`:
```bash
python3 test.py  --vocab_path='data/vocab.pkl' --model_path="models/CAP-35.ckpt"  --caption_path='data/annotations/karpathy_split_test.json'  --result_path='results'  --image_path='coco_resized'  --result_name='CAP' --eval=1 --beam_size=3  --device=0  --model_device=9   --model_name='CAP' --image_size=512 
```
This will generate the result file `results/CAP_captions_val2014_fakecap_results.json`. And run the following command to get scores:
```bash
python3 eval.py --result_file='CAP_b3_captions_val2014_fakecap_results.json' --ann_file="data/annotations/karpathy_split_test.json" --result_path='results'
```
- Test with ensemble models:
```bash
python3 ensemble_test.py  --num_models=4  --vocab_path='data/vocab.pkl' --model_path="models/"  --caption_path="data/annotations/karpathy_split_test.json"  --result_path='results'  --image_path='coco_resized'  --result_name='CAP'  --beam_size=3  --device=0  --model_name="CAP" --save_name="CAP" --image_size=512  --model_device 9 9 9 9
```
This will load 4 models all trained with the main device 9. Please rename 4 models to `CAP1`,...,`CAP4` before running this command.
### Flickr30k
Same as mentioned before, change `dataset`, `image_path`, `caption_path`, and `vocab_path` arguments for flickr30k dataset.