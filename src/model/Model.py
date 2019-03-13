from model.HelperFunctions import convolutional_layer
from src.model.PlaceholderGenerator import generate_placeholders

class Model:

    def __init__(self):
        self.convo_1 = []
        self.convo_1_pooling = []
        self.convo_2 = []
        self.convo_2_pooling = []
        self.convo_2_flat = []
        self.full_layer_one = []
        self.full_one_dropout = []
        self.y_pred = []
