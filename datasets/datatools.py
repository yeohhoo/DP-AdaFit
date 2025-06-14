import pickle
import os
import numpy as np
from PIL import Image

# import lasagne

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_databatch_val(data_file, img_size=32):
    """using python file. Convert the

    Args:
        data_folder (_type_): _description_
        img_size (int, optional): _description_. Defaults to 32.

    Returns:
        _type_: _description_
    
    https://github.com/PatrykChrabaszcz/Imagenet32_Scripts/blob/master
    """
    print("Starting loading images...")
    # data_file = os.path.join(data_folder, 'train_data_batch_')
    
    # xs = []
    # ys = []
    # mean_images = []
    d = unpickle(data_file)
    x = d['data']
    y = d['labels']
    # mean_image = d['mean']
    # xs.append(x)
    # ys.append(y)
    # mean_images.append(mean_image)

    # x = np.concatenate(xs)
    # y = np.concatenate(ys)
    # mean_image = np.concatenate(mean_images)

    # Labels are indexed from 1, shift it so that indexes start at 0
    y = [i-1 for i in y]

    img_size2 = img_size * img_size

    x = np.dstack((x[:, :img_size2], x[:, img_size2:2*img_size2], x[:, 2*img_size2:]))
    x = x.reshape((x.shape[0], img_size, img_size, 3))#.transpose(0, 3, 1, 2)

    y = np.array(y, dtype='int32')

    return dict(
        data=x,#lasagne.utils.floatX(X_train),
        label=y)

def save_images(dataset, out_dir):
    """
    saving .jpg from .npz
    """
    print(f"saving image to {out_dir}...")
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    
    images = dataset['arr_0']
    for i in range(images.shape[0]):
        filename = os.path.join(out_dir, f"{i+50000:08d}.png")
        Image.fromarray(images[i]).save(filename)

def jpg2npz(img_dir, save_dirs):
    """
    It is mainly used to convert .jpg(png) from dataset to .npz with key 'arr_0'
    For example:
        img_dir = '/users/u2020010337/guided-diffusion/mnist_train'
        save_dirs = '/users/u2020010337/guided-diffusion/mnist_train_npz'
        jpg2npz(img_dir, save_dirs)
    img_dir is a directory that contains many image files.
        img_dir/000.jpg
        img_dir/001.jpg
        img_dir/002.jpg
        ....
    """
    all_images = []
    print(f"starting save .npz to {save_dirs} ")

    if not os.path.exists(save_dirs):
        os.mkdir(save_dirs)
    num = 0
    for root, dirs, files in os.walk(img_dir):
        # for file in files:
            # imgs = np.array(Image.open(os.path.join(root, file)).convert("RGB"))
            # all_images.extend([imgs])

        for dir in dirs:
            img_childdir = os.path.join(root, dir)
            for child_root, child_dirs, child_files in os.walk(img_childdir):
                for child_file in child_files:
                    imgs = np.array(Image.open(os.path.join(child_root, child_file)).convert("RGB"))
                    all_images.extend([imgs])

    # print(f"num={num}")
    arr = np.array(all_images)  # NHWC
    
    shape_str = "x".join([str(x) for x in arr.shape])
    out_path = os.path.join(save_dirs, f"{img_dir[-17:]}_{shape_str}.npz")
    print(f"saving to {out_path}")
    
    np.savez(out_path, arr)
    print("saving complete")


def index2label(filename):
    """Convert the index to the corresponding label
    Args:
        filename (.txt): the directory path of documents storing indexes and
                         labels. The content is arranged in this way:
                             n02119789 1 kit_fox
                             n02100735 2 English_setter
                             n02110185 3 Siberian_husky
                             ...
                        eg:
                           idx_to_label = index2label(
                           "H:\dataset\imagenet\map_clsloc.txt"
                           )
    Returns:
        dict: a dictionary which has key with index and value with label.
              arranged in this way:
              {'1': 'kit_fox',
               '2': 'English_setter',
               '3': 'Siberian_husky', ...}
    """
    labelMat = []
    indexMat = []
    fr = open(filename)
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.strip().split()
        num = np.shape(listFromLine)[0]
        labelMat.append(listFromLine[num-1])
        indexMat.append(listFromLine[num-2])
    
    idx_to_label = {indexMat[i]:labelMat[i] for i in range(len(indexMat))}
    return idx_to_label


if __name__ == "__main__":
    # data = np.load(r"/users/u2020010337/guided-diffusion/checkpoints/openai-2024-09-16-13-27-30-632905/samples_50000x32x32x3.npz")
    # out_dir = "/users/u2020010337/guided-diffusion/checkpoints/openai-2024-09-18-20-12-20-410798/images"
    # save_images(data, out_dir)

    # img_dir = '/users/u2020010337/guided-diffusion/cifar10_train'
    # save_dirs = '/users/u2020010337/guided-diffusion/cifar10_train_npz'
    # jpg2npz(img_dir, save_dirs)
    #
    # img_dir = '/users/u2020010337/generated_dataset/synthetic_cifar10'
    # save_dirs = '/users/u2020010337/generated_dataset/synthetic_cifar10_npz'
    # jpg2npz(img_dir, save_dirs)

    # # data_folder = "/hy-tmp/Imagenet32_train"
    # # data = load_databatch(data_folder)
    # # label_dic = index2label("/hy-tmp/map_clsloc.txt")
    # # out_dir = r"/hy-tmp/Imagenet32_train_images"
    # # save_images(data, out_dir, label_dic)
    # data_folder = r"H:\dataset\imagenet\imagenet_32\Imagenet32_val\val_data"
    # data = load_databatch_val(data_folder)
    # shape_str = "x".join([str(x) for x in data['data'].shape])
    # out_dir = r"H:\dataset\imagenet\imagenet_32"
    # out_path = os.path.join(out_dir, f"samples_{shape_str}.npz")
    # # logger.log(f"saving to {out_path}")
    # if data['label'].size!=0:
    #     np.savez(out_path, data['data'], data['label'])
    # else:
    #     np.savez(out_path, data['data'])
    # # label_dic = index2label("H:\dataset\imagenet\imagenet_32\map_clsloc.txt")
    # # out_dir = r"H:\Imagenet32_val_images"
    # # save_images(data, out_dir, label_dic)


    import numpy as np
    data = np.load("/users/u2020010337/generated_dataset/synthetic_cifar10_npz/synthetic_cifar10_50000x32x32x3.npz")
