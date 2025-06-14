# Evaluations

To compare different generative models, we use FID, sFID, Precision, Recall, and Inception Score. These metrics can all be calculated using batches of samples, which we store in `.npz` (numpy) files.

## Reference batches
In the `.npz` file, which contains a dictionaty with keys `["arr_0", "arr_1"]` or `["arr_0"]`. It depends on the dataset whether or not has labels. Assuming it contain 5,000 images with 32*32 which will be used to compute statistic, the `.npz` file will be as follow:
`dict["arr_0"].shape = (5000, 32, 32, 3)`. It represents the shape attribute of the images in the dataset.
`dict["arr_1"].shape = (5000, )`. It represents the shape attribute of the labels in the dataset.

## Sample batches
Same content format as reference batches

# Run evaluations

First, generate or download a batch of samples and download the corresponding reference batch for the given dataset. For this example, we'll use ImageNet 256x256, so the refernce batch is `VIRTUAL_imagenet256_labeled.npz` and we can use the sample batch `admnet_guided_upsampled_imagenet256.npz`.

Next, run the `evaluator.py` script. The requirements of this script can be found in [requirements.txt](requirements.txt). Pass two arguments to the script: the reference batch and the sample batch. The script will download the InceptionV3 model used for evaluations into the current working directory (if it is not already present). This file is roughly 100MB.

The output of the script will look something like this, where the first `...` is a bunch of verbose TensorFlow logging:

```
$ python evaluator.py VIRTUAL_imagenet256_labeled.npz admnet_guided_upsampled_imagenet256.npz
...
computing reference batch activations...
computing/reading reference batch statistics...
computing sample batch activations...
computing/reading sample batch statistics...
Computing evaluations...
Inception Score: 215.8370361328125
FID: 3.9425574129223264
sFID: 6.140433703346162
Precision: 0.8265
Recall: 0.5309
```
## cifar10
we need to prepare the orignal dataset for reference.
### Prepare 
Firstly, saving the `torchvision.datasets.CIFAR10` as `.jpg(.png)` we can use the `./datasets/cifar10.py`. Then, convert the `.jpg` to `.npz`. We can use the `jpg2npz()`in the`./datasets/datatools.py` 
### Run evalations
```
PYTHONPATH=your_root/DP-AdaFit python evaluations/evaluator_dp.py original_dataset.npz  systhetic.npz
```

