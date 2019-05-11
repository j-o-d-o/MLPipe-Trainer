"""
Show images that were saved in to the mongoDB database (after executing mongo_uploader.py)
Working directory is expected to be mlpipe root folder
"""
from mlpipe.data_reader.mongodb import MongoDBConnect
from mlpipe.utils import Config

import numpy as np
import cv2


if __name__ == "__main__":
    Config.add_config('./projects/cifar10/config.ini')
    mongo_con = MongoDBConnect()
    mongo_con.add_connections_from_config(Config.get_config_parser())
    collection = mongo_con.get_collection("localhost_mongo_db", "cifar10", "train")

    documents = collection.find({})
    for doc in documents:
        png_binary = doc["img"]
        png_img = np.frombuffer(png_binary, np.uint8)
        mat_img = cv2.imdecode(png_img, cv2.IMREAD_COLOR)
        cv2.imshow("test", mat_img)
        print(doc["label_name"])
        cv2.waitKey(0)
