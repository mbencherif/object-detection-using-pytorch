import io
import json
import flask
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn

import libs.model_utils as model_utils
import libs.plot_utils as plot_utils
from libs.custom_layers import Flatten
from PIL import Image

# Initialize our Flask application and the PyTorch model.
app = flask.Flask(__name__)
model = None
use_gpu = False

def load_model():
    """Load the pre-trained model, you can use your model just as easily.

    """
    global model
    custom_head = nn.Sequential(
        Flatten(),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512*7*7, 256),
        nn.ReLU(),
        nn.BatchNorm1d(256),
        nn.Dropout(0.5),
        nn.Linear(256, 4+20)
    )

    model = model_utils.get_resnet34_model_with_custom_head(custom_head)
    model.load_state_dict(torch.load('combined_model_val_77.5.ckpt', map_location='cpu'))

    model.eval()
    if use_gpu:
        model.cuda()
        


def get_category_to_label(id):
    id_to_cat = {
        0: 'car',
        1: 'horse',
        2: 'person',
        3: 'aeroplane',
        4: 'train',
        5: 'dog',
        6: 'chair',
        7: 'boat',
        8: 'bird',
        9: 'pottedplant',
        10: 'cat',
        11: 'sofa',
        12: 'motorbike',
        13: 'tvmonitor',
        14: 'bus',
        15: 'sheep',
        16: 'diningtable',
        17: 'bottle',
        18: 'cow',
        19: 'bicycle'}
    return id_to_cat[id]

def test_model_on_img(im):
    sz = 224
    test_tfms = transforms.Compose([
        transforms.Resize((sz, sz)),
        transforms.ToTensor()
    ])
    test_im_tensor = test_tfms(im)[None]
    
    pred_bbox, pred_cat_id = model_utils.test_on_single_image(test_im_tensor, model, sz)
    return plot_utils.get_result_on_test_image(pred_bbox, pred_cat_id, get_category_to_label, im)

@app.route("/predict", methods=["POST"])
def predict():
    # Initialize the data dictionary that will be returned from the view.
    data = {"success": False}

    # Ensure an image was properly uploaded to our endpoint.
    if flask.request.method == 'POST':
        if flask.request.files.get("image"):
            # Read the image in PIL format
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))

            output = test_model_on_img(image)
            response = flask.make_response(output.getvalue())
            response.mimetype = 'image/png'
            return response

            # Indicate that the request was a success.
            data["success"] = True

    # Return the data dictionary as a JSON response.
    return flask.jsonify(data)

if __name__ == '__main__':
    print("Loading PyTorch model and Flask starting server ...")
    load_model()
    app.run()
