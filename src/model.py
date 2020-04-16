from neural_network.utils.voc_annotation import voc
from neural_network.predict import predict
from neural_network.train import train
def voc_deal():
    voc()

def predict_start(img):
    predict(img)

def train_start():
    train()