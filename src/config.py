from torch import cuda


# GENERAL
level = "hard"      # easy/medium/hard
device = "cuda" if cuda.is_available() else "cpu"

#DATA
batch_size = 32

#ENCODING
encode = True

#TRAINING
epochs = 10
lr = 1e-3

# MODEL
model = "mlp"       # "mlp" or "cnn"