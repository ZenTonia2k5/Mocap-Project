import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from mmpose.apis import MMPoseInferencer

models = MMPoseInferencer.list_models('mmpose')
keys = []
if isinstance(models, dict) and 'models' in models:
    for item in models['models']:
        # sometimes it returns a dict or list
        keys.append(str(item))

import json
with open('mmpose_models.json', 'w') as f:
    json.dump(keys, f)
