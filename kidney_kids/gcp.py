import os

from google.cloud import storage

BUCKET_NAME = 'kidney_disaese'
MODEL_NAME = 'forest_model'
MODEL_VERSION = 'v1'


def storage_upload(model_name, rm=False):
    MODEL_NAME = model_name
    client = storage.Client().bucket(BUCKET_NAME)


    local_model_name = 'model.joblib'
    storage_location = f"models/{MODEL_NAME}/{MODEL_VERSION}/{local_model_name}"
    blob = client.blob(storage_location)
    blob.upload_from_filename('model.joblib')
    if rm:
        os.remove('model.joblib')
