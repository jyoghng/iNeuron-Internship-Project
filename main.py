from wsgiref import simple_server
from flask import Flask, request, render_template
from flask import send_file
from flask import Response
import os
from flask_cors import CORS, cross_origin
import flask_monitoringdashboard as dashboard
from upload_file.uploadFile import UploadFile
from data_preprocessing.dataPreprocessing import Preprocessor
from data_validation.dataValidation import DataValidation
from prediction.predictFromModel import Prediction
from training.trainingModel import Training

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
dashboard.bind(app)
CORS(app)

@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')

@app.route("/upload", methods=['POST'])
@cross_origin()
def upload():
    if request.method == 'POST':
        try:
            if 'file' not in request.files:
                return Response('No file selected for uploading')

            file = request.files['file']

            # Upload file Object Initialisation
            upld_file = UploadFile(file)
            # Calling the Upload file function
            resp_msg_upld = upld_file.upload_file(file)
            return Response(resp_msg_upld)

        except Exception as e:
            return Response("Error occured in uploading the file.", e)

    else:
        return render_template('index.html')


@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRouteClient():

    if request.method == 'POST':
        try:
            filepath_val = "Input_Files/Pred_file.csv"
            process_type = "P"

            # Input File Validation Object Initialisation
            pred_valid = DataValidation(filepath_val,process_type)
            # Calling the Data Validation function
            resp_msg_vald = pred_valid.data_validation(filepath_val,process_type)

            if resp_msg_vald == "Validation Success":
                filepath = "Prediction_Files/Good_Raw/Pred_file.csv"
                process_type = "P"
                # Pre-Processor Object Initialisation
                pred_prep = Preprocessor(filepath,process_type)
                # Calling the Data Pre-Processing function
                resp_msg_prep = pred_prep.data_preprocess(filepath,process_type)

                if resp_msg_prep == "Pre-Processing Success":
                    # Prediction Object Initialisation
                    pred = Prediction()
                    # Predicting for the dataset uploaded by the user
                    resp_msg_pred = pred.predict_model()
                    return resp_msg_pred
                else:
                    #return json.dumps(resp_msg_prep)
                    return resp_msg_prep


            else:
                #return json.dumps(resp_msg_vald)
                return resp_msg_vald

        except Exception as e:
            return Response("Error occured during prediction", e)

    else:
        return render_template('index.html')


@app.route("/train", methods=['POST'])
@cross_origin()
def trainRouteClient():
    if request.method == 'POST':

        try:
            filepath_val = "Input_Files/Train_file.csv"
            process_type = "T"

            # Input File Validation Object Initialisation
            train_valid = DataValidation(filepath_val, process_type)
            # Calling the Data Validation function
            resp_msg_vald = train_valid.data_validation(filepath_val, process_type)

            if resp_msg_vald == "Validation Success":
                filepath = "Training_Files/Good_Raw/Train_file.csv"
                process_type = "T"
                # Pre-Processor Object Initialisation
                train_prep = Preprocessor(filepath,process_type)
                # Calling the data_preprocess function
                train_prep.data_preprocess(filepath,process_type)

                # Training Object Initialisation
                train = Training()
                # Training for the dataset uploaded by the user
                resp_msg_train = train.train_model()
                return Response(resp_msg_train)
            else:
                return Response(resp_msg_vald)


        except Exception as e:
            return Response("Error occured during re-training", e)
    else:
        return render_template('index.html')

@app.route('/downloads/')
@cross_origin()
def file_downloads():
        try:
            return send_file('Predicted_Files/Result.csv', attachment_filename='Result.csv',as_attachment=True, cache_timeout=0)
        except Exception as e:
            return Response("Error occured during downloading result", e)

@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r

port = int(os.getenv("PORT",5000))
if __name__ == "__main__":
    host = '0.0.0.0'
    #port = 5000
    httpd = simple_server.make_server(host, port, app)
    # print("Serving on %s %d" % (host, port))
    httpd.serve_forever()
