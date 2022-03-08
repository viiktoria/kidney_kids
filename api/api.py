'''this is the api file, here are our endpoints'''
from codecs import BufferedIncrementalDecoder
from fastapi import FastAPI
from kidney_kids import data
from google.cloud import storage
import joblib
from kidney_kids.gcp import BUCKET_NAME
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from urllib.parse import urlparse
from urllib.parse import parse_qs
from urllib import request
import inspect

from kidney_kids.scatters import scatter, confusion_score
from starlette.responses import StreamingResponse
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


app = FastAPI()
buffi = BytesIO()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


@app.get("/")
async def root():

    return {"message": "something else"}



@app.get("/model")
def confusion_matrix(model, params):
    '''takes in the model and its parameters
    and returns the according confusion matrix'''

    if model == 'knn':
        k = params['k']
        model = KNeighborsClassifier(n_neighbors=k)
    elif model == 'logreg':
        penalty = params['penalty']
        C = params['C']
        model = LogisticRegression(penalty=penalty, C=C)
    else:

        model = RandomForestClassifier()

    #get confusion matrxi from scatters.py
    df_conf_matrix = confusion_score(model)


    return {'conf_matrix': df_conf_matrix.to_json()}


@app.get("/scatter")
def plots(feat_1, feat_2):
    df_plot = scatter(feat_1, feat_2)

    #wie kann man plots zurück geben?
    #return StreamingResponse(plot, media_type="image/png")
    return {'scatter': df_plot.to_json()}

@app.get("/predict")
def predict(age, bp, sg, al, su, rbc, pc, pcc, ba, bgr, bu,
       sc, sod, pot, hemo, pcv, wc, rc, htn, dm, cad,
       appet, pe, ane):
    #parameter list:
    #age, bp, sg, al, su, rbc, pc, pcc, ba, bgr, bu, sc, sod, pot, hemo, pcv, wc, rc, htn, dm, cad, appet, pe, ane
    #this function returns the prediction and probability of the
    #choosen model and features


    ###---this method would be better to get the params to a dict, but its not working:---

    #args = inspect.getfullargspec(predict)
    #this step gives internal error: nontype is not iterable:
    #arg_val_list = list(args.defaults)

    ### make df out of columnnames and values
    #for i,j in zip(args_list, arg_val_list):
    #    dict[i] = [j]
    #df = pd.DataFrame.from_dict(dict)



    args_list = ['age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr', 'bu',
       'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc', 'htn', 'dm', 'cad',
       'appet', 'pe', 'ane']

    BUCKET_NAME = 'kidney_disaese'
    bucket=BUCKET_NAME
    MODEL_NAME = 'forest_model'
    client = storage.Client().bucket(bucket)

    storage_location = 'models/forest_model/v1/model.joblib'
    blob = client.blob(storage_location)
    blob.download_to_filename('model.joblib')
    model = joblib.load('model.joblib')


    ### preprocessing
    #for the preprocesing a df is needed
    dict ={'age': [age],  'bp': [bp], 'sg': [sg], 'al': [al], 'su': [su], 'rbc':[rbc], 'pc': [pc], 'pcc': [pcc], 'ba': [ba], 'bgr': [bgr], 'bu': [bu],
       'sc': [sc], 'sod':[sod], 'pot': [pot], 'hemo': [hemo], 'pcv': [pcv], 'wc': [wc], 'rc': [rc], 'htn': [htn], 'dm': [dm], 'cad': [cad],
       'appet': [appet], 'pe': [pe], 'ane': [ane]}
    df_predict = pd.DataFrame.from_dict(dict)
    X_test = data.get_preproc_data(df_predict)

    ### prediction
    #this format (np.array) is needed for the prediction:
    #array_test = np.array([0.75, 0., 0., 0., 0., 0., 0., 0., 0., 0., 2., 0., 0., 0.29885057, 0.23076923, 0.13034188, 0.12195122, 0.00529801, 0.84858044, 0.01797753, 0.97692308, 0.65, 0.31818182, 0.44067797]).reshape(1, -1)

    result = model.predict(X_test)
    proba = model.predict_proba(X_test)

    return {"result": str(result[0]), "proba": str(proba)}


    #if rm:
    #    os.remove('model.joblib')
    #return model
