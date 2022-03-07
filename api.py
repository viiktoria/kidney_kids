'''this is the api file, here are our endpoints'''
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

'''
@app.get("/model")
def model(model, *params):
    #this function returns the scaterplot and the
    #confusion matrix according to the choosen model (logreg, forest, knn) and the choosen parameters

    #knnmodel
    if model == 'Knn':
        pass

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


'''@app.get("/predict")
def predict(model, features):
    #this function returns the prediction and probability of the
    #choosen model and features

    #somehow like this it has to be done:
    model = joblib.load('model.joblib')
    result = model.predict(preproc(X_predict))

    ###maybe we have to use this function? download from storage to use on google run?
    def download_model(model_directory="PipelineTest", bucket=BUCKET_NAME, rm=True):
    client = storage.Client().bucket(bucket)

    storage_location = 'models/{}/versions/{}/{}'.format(
        MODEL_NAME,
        model_directory,
        'model.joblib')
    blob = client.blob(storage_location)
    blob.download_to_filename('model.joblib')
    print("=> pipeline downloaded from storage")
    model = joblib.load('model.joblib')
    if rm:
        os.remove('model.joblib')
    return model'''
