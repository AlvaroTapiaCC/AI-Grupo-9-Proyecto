from torch import cuda


# GENERAL
device = "cuda" if cuda.is_available() else "cpu"
level = "easy"                  # easy/medium/hard

# MODEL
model = "mlp"                   # "mlp" or "cnn"
train_new = False               # if True, train new model, else use best.pt

#ENCODING
encode = False

#TRAINING
batch_size = 32
epochs = 10
lr = 1e-3
image_size = (64, 64)           # Image size for CNN