import torch
import os
import tensorflow as tf
# os.environ['LD_LIBRARY_PATH'] = '/home/lev/Downloads/cudnn-linux-x86_64-8.9.2.26_cuda11-archive/lib'

# Clear GPU memory
tf.keras.backend.clear_session()

# For PyTorch:
torch.cuda.empty_cache()

print("using cudnn", torch.backends.cudnn.version())