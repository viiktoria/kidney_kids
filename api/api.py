'''this is the api file, here are our endpoints'''
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

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


@app.get("/")
async def root():
    return {"message": "Hello World"}

'''
@app.get("/model")
def model(*params):
    #this function returns the scaterplot and the
    #confusion matrix according to the choosen model (logreg, forest, knn) and the choosen parameters

    #knnmodel
    if model == 'Knn':


    #logregmodel
    elif model == 'LogReg':
        if params['penalty'] == 'l1':
            pass
        elif params["penalty"] == 'l2':
            pass
        else:
            pass
    #forestmodel
    else:
        if params['max_depths'] == 1:
            pass
        else:
            pass'''




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
    X_test = data.preproc(df_predict)

    ### prediction
    #this format (np.array) is needed for the prediction:
    #array_test = np.array([0.75, 0., 0., 0., 0., 0., 0., 0., 0., 0., 2., 0., 0., 0.29885057, 0.23076923, 0.13034188, 0.12195122, 0.00529801, 0.84858044, 0.01797753, 0.97692308, 0.65, 0.31818182, 0.44067797]).reshape(1, -1)
    result = model.predict(X_test)
    proba = model.predict_proba(X_test)

    return {"result": str(result[0]), "proba": proba}


    #if rm:
    #    os.remove('model.joblib')
    #return model
