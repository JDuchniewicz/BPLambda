import joblib
import base64
import numpy as np
import json

from io import BytesIO
from PIL import Image

model_file = '/opt/ml/model'
model = joblib.load(model_file)


def lambda_handler(event, context):
    ppg_bytes = event['body'].encode('utf-8')
    raw_ppg = BytesIO(base64.b64decode(ppg_bytes))

    x = np.array(raw_ppg)
    prediction = model.predict(x.reshape(1, -1))[0]
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
