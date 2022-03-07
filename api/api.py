'''this is the api file, here are our endpoints'''
from fastapi import FastAPI
from kidney_kids import data
from google.cloud import storage
import joblib
from kidney_kids.gcp import BUCKET_NAME
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware

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
def predict(features):
    #this function returns the prediction and probability of the
    #choosen model and features
    BUCKET_NAME = 'kidney_disaese'
    bucket=BUCKET_NAME
    MODEL_NAME = 'forest_model'
    client = storage.Client().bucket(bucket)

    '''storage_location = 'models/{}/versions/{}/{}'.format(
        MODEL_NAME,
        model_directory,
        'model.joblib')'''

    storage_location = 'models/{}/v1'.format('model.joblib')
    blob = client.blob(storage_location)
    blob.download_to_filename('model.joblib')
    model = joblib.load('model.joblib')

    #make dataframe out of paramterdic for preproc-fct:
    features = data.get_cleaned_data[0].head(1)
    df_predict = pd.DataFrame(features)
    X_test = data.preproc(df_predict)

    #prediction
    result = model.predict(X_test)
    proba = model.predict_proba(X_test)

    return result, proba

    #if rm:
    #    os.remove('model.joblib')
    #return model
