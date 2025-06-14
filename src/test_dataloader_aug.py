import argparse
import sys
print(sys.path)
from utils.dataloader_aug import prepare_dataloaders
from utils.dataset import populate_dataset


parser = argparse.ArgumentParser(description="PyTorch imagenet DP Training")
parser.add_argument("--transform",type=int,default=8,help="Order of the Augmentation multplicity (AugMult). If non 0, each image in each batch will be duplicated 'transform' times, and randomly applied 'type_of_augmentation'.",)
parser.add_argument("--type_of_augmentation", default="OursBest", help="type of AugMult that will be used if 'args.transform' is non 0.")
parser.add_argument("--train_transform", choices=["random", "flip", "center", "simclr", "beit","resize"], default="center", help="equivalent to AugMult of order 1, i.e classic data augmentation.")
parser.add_argument("--dataset",type=str,default="imagenet",help="Path to train data",)
parser.add_argument("--num_classes",default=-1,type=int,help="Number of classes in the data set.",)
parser.add_argument("--img_size", type=int, default=None)
parser.add_argument("--crop_size", type=int, default=None)
parser.add_argument("--batch_size", default=256,type=int,metavar="B",help=r"Batch size for simulated training. It will automatically set the noise $\sigma$ s.t. $B/\sigma = B_{ref}/sigma_{ref}$",)

parser.add_argument("--proportion",default=1,type=float,help="Training only on a subset of the training set. It will randomly select proportion training data",)

parser.add_argument("--test_batch_size",default=512,type=int,help="What batch size to use when evaluating the model the test set",)

# ## Data loading related
parser.add_argument( "--train_path",type=str,default="H:\\dataset\\imagenet\\imagenet\\train",help="name of training set")
parser.add_argument("--val_path",type=str,default="H:\\dataset\\imagenet\\imagenet\\val", help="path to validation data",)

#DEIT
####FOR TRANSFORMERS
parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
parser.add_argument('--drop_block', type=float, default=None, metavar='PCT',
                        help='Drop block rate (default: None)')
parser.add_argument('--color_jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
parser.add_argument('--train_interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')
parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
parser.add_argument('--remode', type=str, default='pixel',
                    help='Random erase mode (default: "pixel")')
parser.add_argument('--recount', type=int, default=1,
                    help='Random erase count (default: 1)')
parser.add_argument('--resplit', action='store_true', default=False,
                    help='Do not random erase first (clean) augmentation split')

args = parser.parse_args()

populate_dataset(args)
train_loader,test_loader = prepare_dataloaders(args)
image, target = next(iter(train_loader))
print(f"image[1].shape:{image[1].shape}")

# import numpy as np
# import matplotlib.pyplot as plt


# for j in range(10):
#      fig = plt.figure()
#      for i in range(image[0].shape[0]):
#           img = image[j][i]
#           img = img.numpy()
#           img = np.transpose(img, (1, 2, 0)) # C*H*W -> H*W*C
#           fig.add_subplot(1,8,int(i+1))
#           plt.imshow(img)

#      plt.show()