import joblib
import base64
import numpy as np
import json
import boto3

from io import BytesIO
from PIL import Image

from data_preparation import prepare_data, normalize_data

model_file = '/opt/ml/model'
model = joblib.load(model_file)


def lambda_handler(event, context):
    print(event)
    stringized = str(event).replace('\'', '"')
    ppg = json.loads(stringized)['data']

    ppg = np.array(ppg)
    ppg = normalize_data(ppg)
    prepared = prepare_data(ppg)

    prediction = model.predict(prepared.reshape(1, -1))[0]
    sbp = prediction[0]
    dbp = prediction[1]
    print(f"Predictions {sbp} {dbp}")

    pred_json = json.dumps(
            {
                "SBP": sbp,
                "DBP": dbp,
            })
    client = boto3.client('iot-data')

    response = client.publish(topic='bibop/incoming',
                              qos=0,
                              payload=pred_json)
