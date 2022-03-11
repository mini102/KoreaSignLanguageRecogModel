import numpy as np
import os
import tensorflow as tf

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
actions = ['Oui', 'Bonjour', 'Non','Cava','Silvous']

Tpath = 1645425316
data = np.concatenate([
    np.load(f'dataset/seq_Oui_{Tpath}.npy'),
    np.load(f'dataset/seq_Bonjour_{Tpath}.npy'),
    np.load(f'dataset/seq_Non_{Tpath}.npy'),
    np.load(f'dataset/seq_Cava_{Tpath}.npy'),
    np.load(f'dataset/seq_Silvous_{Tpath}.npy')
], axis=0)

x_data = data[:, :, :-1]
labels = data[:, 0, -1]

print(x_data.shape)
print(labels.shape)

from tensorflow.keras.utils import to_categorical

y_data = to_categorical(labels, num_classes=len(actions))
y_data.shape

from sklearn.model_selection import train_test_split

x_data = x_data.astype(np.float32)
y_data = y_data.astype(np.float32)

x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.1, random_state=2021)

print(x_train.shape, y_train.shape)  #(1144,20,99), (1144,3)
print(x_val.shape, y_val.shape)   #(128,20,99) , (128,3)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard

log_dir = 'logs/newInput/' #'logs/gradient_tape/' 'logs/after/' 
tensorboard_callback = TensorBoard(log_dir=log_dir)
test_summary_writer = tf.summary.create_file_writer(log_dir)

model = Sequential([
    LSTM(64, return_sequences=True, activation='relu', input_shape=x_train.shape[1:3]),  #input shape: 30,99 (30 = qindow의 크기, 99 = landmark, visibility, 각도 등 )
    LSTM(128, return_sequences=True, activation='relu'),  
    LSTM(64, return_sequences=False,activation='relu'),
    Dense(64, activation='relu'),  #Dense의 노드: 64
    Dense(32, activation='relu'), 
    Dense(len(actions), activation='softmax')  #마지막에는 5, actions의 개수만큼 각 actions에 대한 확률 len(actions)
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
model.summary()

from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

history = model.fit(
    x_train,
    y_train,
    validation_data=(x_val, y_val),
    epochs=200,
    callbacks=[
        ModelCheckpoint('models/videoInput.h5', monitor='val_acc', verbose=1, save_best_only=True, mode='auto'),  #model 저장
        ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=50, verbose=1, mode='auto'),  #learning rate 조절
        tensorboard_callback
    ]
)

model.save('models/videoInput.h5')
import matplotlib.pyplot as plt

fig, loss_ax = plt.subplots(figsize=(16, 10))
acc_ax = loss_ax.twinx()

loss_ax.plot(history.history['loss'], 'y', label='train loss')
loss_ax.plot(history.history['val_loss'], 'r', label='val loss')
loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
loss_ax.legend(loc='upper left')

acc_ax.plot(history.history['acc'], 'b', label='train acc')
acc_ax.plot(history.history['val_acc'], 'g', label='val acc')
acc_ax.set_ylabel('accuracy')
acc_ax.legend(loc='upper left')

plt.show()

# from sklearn.metrics import multilabel_confusion_matrix
# from tensorflow.keras.models import load_model

# model = load_model('models/model.h5')

# y_pred = model.predict(x_val)

# multilabel_confusion_matrix(np.argmax(y_val, axis=1), np.argmax(y_pred, axis=1))


