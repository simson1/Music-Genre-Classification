import os
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import svm
from werkzeug.utils import secure_filename
from flask import Flask, request,redirect, url_for,Response
from flask_cors import CORS, cross_origin
from flask_restful import Resource, Api
from json import dumps
from flask_jsonpify import jsonify
from flask import Flask, request, jsonify
from sklearn import svm
from sklearn import datasets
from sklearn.externals import joblib
from keras import backend as K
import os
import time
import h5py
import json
import sys
from tagger_net import MusicTaggerCRNN
from keras.optimizers import SGD
import numpy as np
from keras.utils import np_utils
from math import floor
from music_tagger_cnn import MusicTaggerCNN
from utils import save_data, load_dataset, save_dataset, sort_result, predict_label, extract_melgrams
import matplotlib.pyplot as plt

UPLOAD_FOLDER = 'music'
ALLOWED_EXTENSIONS = set(['mp3'])
app = Flask(__name__)
api = Api(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
CORS(app)
@app.route("/")
def hello():
    return jsonify({'text':'Hello World!'})
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
@app.route('/api/predict', methods=['POST'])
def predict():
    # get iris object from request
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            return 'No file part'
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], 'example.mp3'))
            TEST = 1
            LOAD_MODEL = 0
            LOAD_WEIGHTS = 1
            MULTIFRAMES = 1
            time_elapsed = 0

            # GTZAN Dataset Tags
            tags = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
            tags = np.array(tags)

            # Paths to set
            model_name = "example_model"
            model_path = "models_trained/" + model_name + "/"
            weights_path = "models_trained/" + model_name + "/weights/"

            test_songs_list = 'list_example.txt'


            # Initialize model
            model = MusicTaggerCRNN(weights=None, input_tensor=(1, 96, 1366))

            model.compile(loss='categorical_crossentropy',
                          optimizer='adam',
                          metrics=['accuracy'])

            if LOAD_WEIGHTS:
                model.load_weights(weights_path+'crnn_net_gru_adam_ours_epoch_40.h5')

            #model.summary()

            X_test, num_frames_test= extract_melgrams(test_songs_list, MULTIFRAMES, process_all_song=False, num_songs_genre='')
            print num_frames_test
            num_frames_test = np.array(num_frames_test)

            t0 = time.time()

            print '\n--------- Predicting ---------','\n'
            print X_test.shape[0]
            print num_frames_test
            print tags.shape[0]
            print num_frames_test.shape[0]
            results = np.zeros((X_test.shape[0], tags.shape[0]))
            predicted_labels_mean = np.zeros((num_frames_test.shape[0], 1))
            predicted_labels_frames = np.zeros((X_test.shape[0], 1))

            song_paths = open(test_songs_list, 'r').read().splitlines()

            previous_numFrames = 0
            n=0
            for i in range(0, num_frames_test.shape[0]):
                ldk=[]
                num_frames=num_frames_test[i]
                print 'Num_frames of 30s: ', str(num_frames),'\n'

                results[previous_numFrames:previous_numFrames+num_frames] = model.predict(
                   X_test[previous_numFrames:previous_numFrames+num_frames, :, :, :])

                s_counter = 0
                for j in range(previous_numFrames, previous_numFrames+num_frames):
                    #normalize the results
                    total = results[j,:].sum()
                    results[j,:]=results[j,:]/total
                    print 'Percentage of genre prediction for seconds '+ str(20+s_counter*30) + ' to ' \
                        + str(20+(s_counter+1)*30) + ': '
                    #sort_result(tags, results[j,:].tolist())

                    predicted_label_frames=predict_label(results[j,:])
                    predicted_labels_frames[n]=predicted_label_frames
                    s_counter += 1
                    n+=1
                

                print '\n', 'Mean genre of the song: '
                results_song = results[previous_numFrames:previous_numFrames+num_frames]

                mean=results_song.mean(0)
                result = zip(tags, mean.tolist())
                sorted_result = sorted(result, key=lambda x: x[1], reverse=True)
                for name, score in sorted_result:
                    d={}
                    score = np.array(score)
                    score *= 100
                    d['name']=name
                    d['value']=round(score,3)
                    #print name, ':', '%5.3f  ' % score, '   '
                    ldk.append(d)
            print ldk
            return json.dumps(ldk)
    return ""
@app.route('/api/train', methods=['POST'])
def train():
    # get parameters from request
    parameters = request.get_json()
    data=pd.read_csv('dataset.csv')
    X_train, X_test, y_train, y_test = train_test_split(data.iloc[:, :-1], data.iloc[:, -1], test_size=0.2)
    clf = svm.SVC(gamma=float(parameters['gamma']), C=float(parameters['C']),kernel=parameters['kernel'],degree=parameters['degree'])
    clf.fit(X_train,y_train) 
    predictions=clf.predict(X_test)
    return jsonify({'accuracy': accuracy_score(y_test, predictions)*100})
if __name__ == '__main__':
    app.run(port=5002)