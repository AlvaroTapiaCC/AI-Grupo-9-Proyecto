from torch import cuda


# GENERAL
level = "easy"      # easy/medium/hard
device = "cuda" if cuda.is_available() else "cpu"

#DATA
batch_size = 32

#ENCODING
encode = False

#TRAINING
epochs = 10
lr = 1e-3

# MODEL
model = "cnn"       # "mlp" or "cnn"
train_new = True    # if True, train new model, else use best.pt

# CNN
image_size = (64, 64)