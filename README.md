
# Human Activity Recognition: a comparison of different convolutional neural network architectures
[![License][licence-badge]](/LICENSE)

Research project to explore different convolutional neural networks architectures that can be used for classification purposes in HAR tasks. The implementation is done in Keras.

![HAR prediction demo](https://github.com/damnko/har-convolutional-neural-networks/blob/master/predict-web-app/preview.gif?raw=true "HAR prediction demo")

## Prerequisites
Download the original dataset at  http://www.kn-s.dlr.de/activity/ and place it in the `dataset-orig` folder

## Project structure
* [001-data-exploration.ipynb](https://github.com/damnko/har-convolutional-neural-networks/blob/master/001-data-exploration.ipynb) contains the code to convert the original dataset in Pandas dataframe and save the output in a `.h5` file
* [002-ml.ipynb](https://github.com/damnko/har-convolutional-neural-networks/blob/master/002-ml.ipynb) contains the code to select the datasets and models to train. The model specifications are in the `./models` folder. The python version of this notebook is saved in `ml.py`
* `generate-plots.ipynb` generates `eps` plots from `csv` files exported from Tensorboard. Those files were used in the research paper `project-paper.pdf`
* `ae-stacked.py` contains the code of the stacked denoising convolutional autoencoder
* `ae-test.py` trains and tests a denoising autoencoder and outputs some reference plots for a qualitative assessment of its reconstruction ability
* `utils.py` contains various utilities
* The folder `/model-testing` contains a jupyter notebook used to test the trained model with some real data captured with a smartphone, see section *Testing with real data* for additional details. This folder contains also the final model and the best weights obtained during training.
* The folder `/predict-web-app` contains the demo web app used to test real time prediction on smartphone sensors' data

## Run the code for training
Training was done on a p2.xlarge AWS EC2 instance using "Deep Learning AMI (Amazon Linux) Version 13.0 - ami-095a0c6d4aed8643d" image.

Following are some useful links to get started:

* https://machinelearningmastery.com/develop-evaluate-large-deep-learning-models-keras-amazon-web-services/
* https://hackernoon.com/keras-with-gpu-on-amazon-ec2-a-step-by-step-instruction-4f90364e49ac
* https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ebs-using-volumes.html

Once the instance is running, just `ssh` to it, run `source activate tensorflow_p36` to activate the desired environment and run `python ml.py` to start the training. Prior to this, the datasets have to be created with  [001-data-exploration.ipynb](https://github.com/damnko/har-convolutional-neural-networks/blob/master/001-data-exploration.ipynb) and put in the `/datasets` folder.

When starting the training make sure that GPU is used, a similar notice should appear when running `python ml.py`
    
    Available gpus ['/job:localhost/replica:0/task:0/device:GPU:0']

If nothing appears between `[]` probably no GPU is detected and the code will run on CPU.

## Testing with real data
The folder `/model-testing` contains a jupyter notebook that can be used to test the trained model with some real data, recorded from a smartphone sensor. The `csv` files found in `/model-testing/test-signals` are recorded with the [IMU+GPS](https://play.google.com/store/apps/details?id=de.lorenz_fenster.sensorstreamgps) Android app and have the following structure

    timestamp | id | acc_x | acc_y | acc_z | id | gyr_x | gyr_y | gyr_z | id | mag_x | mag_y | mag_z
 
Sensor data was recorded following the same setup of the data used for training: the smartphone was positioned in the belt of the user, in *portrait* position.
 
### Real time prediction test
To test real time predictions with the trained model, there is a simple web app in the `/predict-web-app` folder. To test it, these steps

 1. Connect the computer and the smartphone to the same network
 2. Identify the IP address of the server (the computer) by using `ifconfig` on Linux or `ipconfig` on Windows
 3. Run `python server/server.py` to run the server
 4. Open `client.html` in your web browser
 5. Start sending the `UDP` stream from the smartphone app to the IP address of the server (using `fast` update frequency)
 6. You should now see the values updated on real time on the browser

*Note: `Chart.js` was used as charting library because it was the only one among the tested ones that was able to update the plots in real time without affecting the performances of the browser*

## Additional info
Other useful information on the project and HAR in general can be found in the project paper `project-paper.pdf` as well as the source latex used to generate it.

## License
Use as you wish. This project is licensed under the MIT License.


[licence-badge]: https://img.shields.io/npm/l/express.svg