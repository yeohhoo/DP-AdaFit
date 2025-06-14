import pickle
import os
import numpy as np
from PIL import Image

# import lasagne

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_databatch(data_folder, img_size=32):
    """using python file. Convert the

    Args:
        data_folder (_type_): _description_
        img_size (int, optional): _description_. Defaults to 32.

    Returns:
        _type_: _description_
    
    https://github.com/PatrykChrabaszcz/Imagenet32_Scripts/blob/master
    """
    print("Starting loading images...")
    data_file = os.path.join(data_folder, 'train_data_batch_')
    
    xs = []
    ys = []
    mean_images = []
    for idx in range(1, 11): #11
        d = unpickle(data_file + str(idx))
        x = d['data']
        y = d['labels']
        mean_image = d['mean']
        xs.append(x)
        ys.append(y)
        mean_images.append(mean_image)

    x = np.concatenate(xs)
    y = np.concatenate(ys)
    mean_image = np.concatenate(mean_images)

    # Labels are indexed from 1, shift it so that indexes start at 0
    y = [i-1 for i in y]

    img_size2 = img_size * img_size

    x = np.dstack((x[:, :img_size2], x[:, img_size2:2*img_size2], x[:, 2*img_size2:]))
    x = x.reshape((x.shape[0], img_size, img_size, 3))#.transpose(0, 3, 1, 2)

    y = np.array(y, dtype='int32')

    return dict(
        data=x,#lasagne.utils.floatX(X_train),
        label=y,
        mean=mean_image)

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

def save_images(dataset, out_dir, label_dic):
    print(f"saving image to {out_dir}...")
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    
    images, labels = dataset['data'], dataset['label']
    for i in range(images.shape[0]):
        filename = os.path.join(out_dir, f"{label_dic[str(labels[i]+1)]}_{str(labels[i])}_{i:08d}.png")
        Image.fromarray(images[i]).save(filename)


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
    # data_folder = "/hy-tmp/Imagenet32_train"
    # data = load_databatch(data_folder)
    # label_dic = index2label("/hy-tmp/map_clsloc.txt")
    # out_dir = r"/hy-tmp/Imagenet32_train_images"
    # save_images(data, out_dir, label_dic)
    data_folder = "/hy-tmp/val_data"
    data = load_databatch_val(data_folder)
    label_dic = index2label("/hy-tmp/map_clsloc.txt")
    out_dir = r"/hy-tmp/Imagenet32_val_images"
    save_images(data, out_dir, label_dic)
