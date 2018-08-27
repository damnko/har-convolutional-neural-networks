# Human Activity Recognition: a comparison of different convolutional neural network architectures

Research project to explore different convolutional neural networks architectures that can be used for classification purposes in HAR tasks. The implementation is done in Keras.

## Prerequisites

Download the original dataset at  http://www.kn-s.dlr.de/activity/ and place it in the `dataset-orig` folder

## Project structure

* `001-data-exploration.ipynb` contains the code to convert the original dataset in Pandas dataframe and save the output in a `h5`
* `002-ml.ipynb` contains the code to select the datasets and models to train. The model specifications are in the `./models` folder. The python version of this notebook is saved in `ml.py`
* `generate-plots.ipynb` generates `eps` plots from `¢sv` files exported from Tensorboard. Those files were used in the research paper `project-paper.pdf`
* `ae-stacked.py` contains the code of the stacked denoising convolutional autoencoder
* `ae-test.py` trains and tests a denoising autoencoder and outputs some reference plots for a qualitative assessment of its reconstruction ability
* `utils.py` contains various utilities

## Run the code

Training was done on a p2.xlarge AWS EC2 instance using "Deep Learning AMI (Amazon Linux) Version 13.0 - ami-095a0c6d4aed8643d" image.

Following are some useful links to...

* https://machinelearningmastery.com/develop-evaluate-large-deep-learning-models-keras-amazon-web-services/
* https://hackernoon.com/keras-with-gpu-on-amazon-ec2-a-step-by-step-instruction-4f90364e49ac
* https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ebs-using-volumes.html

Once the instance is running, just `ssh` to it, run `source activate tensorflow_p36` to activate the desired environment and run `python ml.py` to start the training.

When starting the training make sure that GPU is used, a similar notice should appear when running `python ml.py`
    
    Available gpus ['/job:localhost/replica:0/task:0/device:GPU:0']

If nothing appears between `[]` probably no GPU is detected and the code will run on CPU.

## Additional info
Other useful information on the project and HAR in general can be found in the project paper `project-paper.pdf` as well as the source latex used to generate it.


## License

Use as you wish. This project is licensed under the MIT License.

