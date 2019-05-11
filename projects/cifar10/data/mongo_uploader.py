"""
Upload cifar-10 data to the mongoDB to access it via that mongoDB reader for training and inference
Cifar-10 data is available here: https://www.cs.toronto.edu/~kriz/cifar.html (download the python version)
Working directory is expected to be mlpipe root folder
"""
from mlpipe.data_reader.mongodb import MongoDBConnect
from mlpipe.utils import Config
import cv2
from numba import jit
import pickle
import numpy as np

# adapt directory to the extract cifar-10 folder
BASE_DIR = "/home/user/Downloads/cifar-10-batches-py/"


@jit
def create_img(arr):
    red = arr[:1024]
    green = arr[1024:2048]
    blue = arr[2048:]
    img_mat = np.zeros((32, 32, 3), np.uint8)
    for col in range(0, 32):
        for row in range(0, 32):
            pos = 32*col + row
            img_mat[col][row][0] = blue[pos]
            img_mat[col][row][1] = green[pos]
            img_mat[col][row][2] = red[pos]
    return img_mat


def get_labels():
    with open(BASE_DIR + "batches.meta", 'rb') as file:
        meta_data = pickle.load(file, encoding='bytes')
        return_val = []
        for name in meta_data[b'label_names']:
            return_val.append(name.decode("utf-8"))
        return return_val


def upload_data(collection, file_path, data_labels):
    with open(file_path, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
        np_arr = data[b"data"]

        for x, label_int in enumerate(data[b"labels"]):
            label_str = data_labels[label_int]

            img_raw = np_arr[x]
            img = create_img(img_raw)

            png_img_str = cv2.imencode(".png", img)[1].tostring()

            collection.insert_one({
                'label': label_int,
                'label_name': label_str,
                'img': png_img_str,
                'content_type': "image/png"
            })


if __name__ == "__main__":
    print("Start uploading")

    Config.add_config('./projects/cifar10/config.ini')
    mongo_con = MongoDBConnect()
    mongo_con.add_connections_from_config(Config.get_config_parser())
    coll_train = mongo_con.get_collection("localhost_mongo_db", "cifar10", "train")
    coll_test = mongo_con.get_collection("localhost_mongo_db", "cifar10", "test")

    labels = get_labels()

    # upload train batches in single train collection
    for i in range(1, 6):
        print("Uploading Batch " + str(i))
        file_name = BASE_DIR + "data_batch_" + str(i)
        upload_data(coll_train, file_name, labels)

    # upload test batches in test collection
    print("Uploading Test Batch")
    upload_data(coll_test, BASE_DIR + "test_batch", labels)

    print("Uploading done")
