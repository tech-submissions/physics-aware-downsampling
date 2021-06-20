# Physics Aware Downsampling

This code was used to implement: Physics-Aware Downsampling with Deep Learning for Scalable Flood Modeling (2021)

## Dependencies
```
pip install -r requirements.txt
```

## Data

The data size (+source files) is around ~300GB.
### Downloading data
1. The source files are listed in the **source** directory. Run the following command to download tif source files:
```
wget -i source/001/ned232_20200818_172734.txt -P /home/usgs_dem_data_source/001
wget -i source/002/ned69_20200818_175155.txt -P /home/usgs_dem_data_source/002
wget -i source/003/ned173_20200818_174748.txt -P /home/usgs_dem_data_source/003
```
2. After the tif files are downloaded, run **source.py** (under **source** directory). This will process the source files and construct an index for the dataset.
3. Lastly, run **ground_truth.py** to calculate fine resolution hydraulic solutions. To save time, the calculation is done in parallel. Configure the number of GPUs to use in **ground_truth.py**.

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
