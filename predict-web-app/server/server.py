# generic imports
import pandas as pd
import numpy as np
import time
import socket
import asyncio
import websockets
import json
from keras.models import model_from_json

def ohe_to_label(ohe_labels):
    Y = [np.argmax(t) for t in ohe_labels]
    return Y

class_conversion = {
    '0': 'falling',
    '1': 'jumping',
    '2': 'lying',
    '3': 'running',
    '4': 'sitting',
    '5': 'standing',
    '6': 'walking'
}

# mean and std computed during training
mean = [-8.33711405e+00, 6.74759490e-01, 7.00107768e-01, -1.04606607e-02, 6.75372537e-03, 2.03051039e-02]
std = [4.27451193, 3.23502816, 3.52895212, 0.53593171, 0.42801951, 0.35004746]

# socket setup
port = 5555
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.bind(("", port))

# load model
json_file = open('m_1d2d_01_reg-augmented-model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("m_1d2d_01_reg-augmented-weights.h5")
print("Model is loaded")
loaded_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

print("Waiting on port:", port)

async def watch(websocket, path):
    prediction_arr = []
    framerate_timer = time.time()
    prediction_timer = time.time()
    samples = 0
    framerate = 0
    while True:
        # data received via udp from smartphone app
        data, addr = s.recvfrom(1024)
        # convert binary to array. Data is like [time, acc, gyro, mag]
        #'4065.56081, 3,   0.364,  0.412, 10.236, 4,  -0.004, -0.002,  0.007, 5,  -2.125,  7.500,-18.188'
        data_str = data.decode('ascii')
        data_arr = [float(i) for i in data_str.split(',')]
        # skip if data is incomplete (happens on startup)
        if len(data_arr) < 13:
            print('Skipping incomplete sample')
            continue

        # append current data to prediction array
        prediction_arr.append(data_arr)
        # initialize empty vars
        prediction = ''
        scores = []

        # do a prediction every x seconds
        predict_every = 0.9
        # it time has passed, predict
        if time.time()-prediction_timer > predict_every:
            # DATA PREPROCESSING
            # acc_x and acc_y were inverted to match the format used during training
            cols = ['time', 'id1', 'acc_y', 'acc_x', 'acc_z', 'id2', 'gyr_y', 'gyr_x', 'gyr_z', 'id3', 'magx', 'magy', 'magz']
            usecols = [2,3,4,6,7,8]
            realdf = pd.DataFrame(prediction_arr, columns=cols)
            # using only a subset of all columns
            realdf = realdf.iloc[:, usecols]
            # invert acc_x axis
            realdf['acc_x'] *= -1
            # sort columns in order to have acc_x before acc_y and same for gyr
            realdf = realdf.reindex_axis(sorted(realdf.columns), axis=1)
            # normalize with values used during training
            realdf_norm = (realdf-mean)/std

            # PREDICT
            """
                In situations where the framerate of the sensor is low (in my case it was 50Hz) I found useful to interpolate
                the datapoints to reach 128 frames needed to do the prediction. This has worked better compared to using
                padding. Eg. If the sensor framerate is 50Hz, the following linear interpolation will generate 128-50 frames
                by interpolating the 50 frames captured during 1 second (if predict_every = 1) 
            """
            X = np.array(realdf_norm)
            # how many samples have been stored until now
            samples_stored = X.shape[0]
            # values in which we want to compute the interpolation
            # interpolate from 0 to the number of samples stored, in the end we want 128 frames
            # because that's the value the model was trained with
            x_target = np.linspace(0, samples_stored, 128)
            # nr of original samples
            x_input = np.arange(samples_stored)
            # create the new input array by computing the interpolation along each of
            # the 6 axis of the sensor values
            # there might be more performant solutions, but this works well with small arrays
            X = np.array([np.interp(x_target, x_input, X[:,i]) for i in range(6)]).T
            X = X.reshape(1,1,128,6)

            scores = loaded_model.predict(X)
            print(np.round(scores, 3))

            activity_id = str(ohe_to_label(scores)[0])
            prediction = class_conversion[activity_id]
            print(time.time(), prediction)

            scores = scores.tolist()
            # empty the array
            prediction_arr = []
            # reset the timer for the next prediction
            prediction_timer = time.time()

        # calculate frequency, updated every 1s
        samples += 1
        if time.time()-framerate_timer >= 1:
            #print('Framerate: {} samples-sec'.format(samples))
            framerate = samples
            samples = 0
            # reset start
            framerate_timer = time.time()

        # data to send to client
        client_data = {
            'realtime': data_arr,
            'scores': scores,
            'prediction': prediction,
            'framerate': framerate
        }
        # send to client via websocket
        await websocket.send(json.dumps(client_data, ensure_ascii=False))
        #await asyncio.sleep(0.1)

start_server = websockets.serve(watch, '127.0.0.1', 5678)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()