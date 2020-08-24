## Installation
The code is based on [PointNet](https://github.com/charlesq34/pointnet), [PointNet++](https://github.com/charlesq34/pointnet2), [SpiderCNN](https://github.com/xyf513/SpiderCNN), [PointConv](https://github.com/DylanWusee/pointconv) and [SceneEncoder](https://github.com/azuki-miho/SceneEncoder). Please install [TensorFlow](https://www.tensorflow.org/install/), and follow the instruction in [PointNet++](https://github.com/charlesq34/pointnet2) to compile the customized TF operators in the *tf_ops* directory. Specifically, you may need to check tf_xxx_compile.sh under each *tf_ops* subfolder, and modify ${CUDA_PATH} and the version of python if necessary.

For example, you need to change the shell script *tf_ops/3d_interpolation/tf_interpolate_compile.sh* from
```
CUDA_PATH=/usr/local/cuda
```
to 
```
CUDA_PATH=/usr/local/cuda-9.0
```
if you use CUDA 9.0.

Additionally, you need to specifically change the command
```
TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
```
to 
```
TF_INC=$(python3 -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
```
if your default python is python2.

The code has been tested with Python 3.6, TensorFlow 1.13.1, CUDA 10.0 and cuDNN 7.3 on Ubuntu 18.04.

To install some of required package, run:
```
pip install -r requirements.txt
```

## Usage
### ScanNet DataSet Segmentation
Please download the ScanNetv2 dataset from [here](http://www.scan-net.org/), and see `scannet/README` for details of preprocessing.

To train a model to segment Scannet Scenes:

```
CUDA_VISIBLE_DEVICES=0 python train_scannet_IoU.py --model scene_encoder_rsl --log_dir scannet_ --batch_size 8
```

After training, to generate test results to *dump_%s* directory:

```
CUDA_VISIBLE_DEVICES=0 python evaluate_scannet.py --model scene_encoder_rsl --batch_size 8 --model_path scannet_%s
```

Then, to upload the results to the ScanNetv2 benchmark server:
```
zip out.zip dump_%s/scene*
```

(Optional) To visualize the results on validation dataset:
```
CUDA_VISIBLE_DEVICES=0 python visualize_scene.py --model scene_encoder_rsl --batch_size 8 --model_path scannet_%s --ply_path DataSet/ScanNetv2/scans
```

Modify the model_path to your .ckpt file path and the ply_path to the original [ScanNetv2](http://www.scan-net.org/) ply file.

### S3DIS DataSet Segmentation

Incoming :=)

### ShapeNet DataSet Segmentation

Incoming :=)

## Acknowledgement
Thanks:

## License
This repository is released under MIT License (see LICENSE file for details).
