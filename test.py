import numpy as np
from params import RBMParams, SGDParams
from models.rbm import RBM, RBMTrainer

# RBM AND TRAINER EXPECT PARAMETER OBJECTS
model_params = RBMParams()
train_params = SGDParams()

# NON-DEFAULT MODEL PRAMETERS
new_model_params = {'n_vis': 28*28, 
                    'n_hid': 500}

# UPDATE TRAINING PARAMETERS
model_params.update(new_model_params)
print '-'*5 + 'RBM parameters'+'-'*5
model_params.show()

# NON-DEFAULT TRAINING PRAMETERS
new_training_params = {'n_epoch': 10,
                       'lrate': 0.01,
                       'w_penalty': -0.00001,
                       'verbose': True, 
                       'display_every': 1.,
                       'visualize': True}

# UPDATE TRAINING PARAMETERS
train_params.update(new_training_params)
print ''
print '-'*5+'Training parameters'+'-'*5
train_params.show()

# INITIALIZE RBM MODEL AND TRAINER
model = RBM(model_params)
trainer = RBMTrainer(model, train_params)

print ''
print '-'*5+'RBM Object'+'-'*5
trainer.rbm.show()
print ''
print '-'*5+'RBM Trainer Object'+'-'*5
trainer.show()

import h5py
mnist_file = '/home/dustin/data/featureLearning/MNIST/mnistLarge.mat'
mnist = h5py.File(mnist_file)

data = mnist['testData'].value.T

from preproc import Image
m = Image()
data = m.transform(data)

# TRAIN
rbm, log = trainer.train(data)

from matplotlib import pyplot as plt

plt.figure(1)
plt.clf()
w = rbm.W[:,0]
plt.imshow(w.reshape(28,28),cmap='gray')
plt.show()

plt.figure()
plt.plot(log['error'])
plt.show()

import time
time.sleep(1000)











