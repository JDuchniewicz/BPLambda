import joblib
import base64
import numpy as np
import json

from io import BytesIO
from PIL import Image

from data_preparation import prepare_data

model_file = '/opt/ml/model'
model = joblib.load(model_file)


def lambda_handler(event, context):
    body = event['body'].encode('utf-8')
    ppg = json.loads(body)['data']

    ppg = np.array(ppg)
    prepared = prepare_data(ppg)

    prediction = model.predict(prepared)[0]
    sbp = prediction[0]
    dbp = prediction[1]

    return {
        'statusCode': 200,
        'body': json.dumps(
            {
                "SBP": sbp,
                "DBP": dbp,
            }
        )
    }
