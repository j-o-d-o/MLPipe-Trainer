# Cifar10 Example

Copy and rename the config.ini.template file to config.ini. In case your MongoDB is not locally and/or needs authentication, add it to the config.ini file.</br>
The following steps the working dir to be the MLPipe root and the conda environment being active e.g. with `>> conda activate mlpipe_env`. In case you are trying this example outside of the repository you will need "opencv-python" as an extra dependency. This would be an example environment.yml:
```yml
# environment.yml
name: cifar10_env
dependencies:
  - python=3.6
  - pip
  - pip:
    - mlpipe-trainer
    - opencv-python
```

### Upload Cifar-10 data to a MongoDB
First download the Cifar-10 data (https://www.cs.toronto.edu/~kriz/cifar.html) as the python version.</br>
Adapt the path to the raw cifar-10 data in _examples/cifar10/data/mongo_uploader.py_ and execute it:
```bash
>> python examples/cifar10/data/mongo_uploader.py
```

### Training
To start the training execute
```bash
>> python examples/cifar10/train.py
```
This will train the Cifar-10 model and save the results as well as the epoch to the MongoDB database. It will show the ObjectId of the saved training. Copy this Id for the next step.

### Inference
To use the trained model for inference, use the copied ObjectId from the Training setp and call:
```bash
>> python examples/cifar10/inference.py OBJECT_ID
```