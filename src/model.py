from neural_network.utils.voc_annotation import voc
from neural_network.predict import predict
from neural_network.train import train
def voc_deal():
    voc()

def predict_start(img):
    f_predict=predict(img)
    # predict(img)
    return f_predict

def train_start(Objct):
    train(Objct)