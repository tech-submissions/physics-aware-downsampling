# Physics Aware Downsampling

This code was used to implement: Physics-Aware Downsampling with Deep Learning for Scalable Flood Modeling (2021)

## Dependencies
```
pip install -r requirements.txt
```

## Data

- Configure your dataset path at **data.py**.

## Example

For distributed training with 4 GPUs, batch size of 4 per GPU, and SGD optimization:
```
python -m torch.distributed.launch --nproc_per_node=4 main.py --batch_size 4 --epochs 50 --optimizer sgd
```
For a single sample evaluation, use the simulation mode:
```
python evaluation.py --sample #sample_num --model #path_to_model
```
For a complete test set evaluation:
```
python evaluation.py --model #path_to_model
```
