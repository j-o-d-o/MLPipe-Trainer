from bson import ObjectId
from tensorflow.keras.models import load_model
import gridfs
import os
import numpy as np
import sys
import cv2

from mlpipe.data_reader.mongodb import MongoDBConnect
from projects.cifar10.processor import PreProcessData
from mlpipe.utils import Config

LABELS = [
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

if __name__ == "__main__":
    # ID of the training that should be loaded
    TRAINING_ID = "5cd69eef32b90173f2e915b2"
    # specify epoch the weights should be loaded from, takes latest if None
    EPOCH = None

    if len(sys.argv) > 1:
        TRAINING_ID = sys.argv[1]
    if len(sys.argv) > 2:
        INDEX = int(sys.argv[2])

    Config.add_config('./projects/cifar10/config.ini')
    mongo_con = MongoDBConnect()
    mongo_con.add_connections_from_config(Config.get_config_parser())

    db = mongo_con.get_db("localhost_mongo_db", "models")
    col_models = mongo_con.get_collection("localhost_mongo_db", "models", "training")
    col_data = mongo_con.get_collection("localhost_mongo_db", "cifar10", "test")

    training_obj = col_models.find_one({"_id": ObjectId(TRAINING_ID)})

    # load model weight data as h5 file from mongoDB
    fs = gridfs.GridFS(db)
    idx = -1 if EPOCH is None else EPOCH
    h5_file = fs.get(training_obj["weights"][idx]["model_gridfs"])
    h5_bytes = h5_file.read()

    tmp_filename = "tmp_model_weights_read.h5"
    with open(tmp_filename, 'wb') as f:
        f.write(h5_bytes)

    # create model with custom metric objects as used while training
    model = load_model(tmp_filename)
    os.remove(tmp_filename)

    data_set = col_data.find()

    processor = PreProcessData()
    for row in data_set:
        _, input_data, _, _ = processor.process(row, None, None)
        input_data = np.asarray([input_data])
        prediction = model.predict(input_data)[0]

        # check if prediction is correct
        max_class_idx = int(np.argmax(prediction))
        max_conf = prediction[max_class_idx]
        label = LABELS[max_class_idx]

        print("{} \t\t({}): \t{:.4f}".format(label, max_class_idx, max_conf))

        # show image
        png_binary = row["img"]
        png_img = np.frombuffer(png_binary, np.uint8)
        mat_img = cv2.imdecode(png_img, cv2.IMREAD_COLOR)
        cv2.imshow("Image", mat_img)
        cv2.waitKey(0)

