import matplotlib.pyplot as plt
import numpy as np
from keras_preprocessing import image
import sys
import os
import tensorflow as tf
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dropout, Flatten, Dense, Activation
from tensorflow.python.keras.layers import  Convolution2D, MaxPooling2D
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import applications
from sklearn.metrics import confusion_matrix
import seaborn as sn
from keras.models import load_model
from sklearn.metrics import classification_report

#pip install tf-nightly

#Se descomenta el modelo pre-entrenado donde se guarda en la variable red
# se copia y se reemplaza la linea de preprocess en test_batches dependiendo del modelo 
#Se cambia el tamaño de la imagen de entrada dependiendo del modelo en TRAINING DIR Y test_data, el tamaño esta dado para cada uno

#MOBILENET
#red = tf.keras.applications.mobilenet.MobileNet()
#tf.keras.applications.mobilenet.preprocess_input   //224

#RESNET152
#red = tf.keras.applications.resnet.ResNet152()
#tf.keras.applications.resnet.preprocess_input      //224

#INCEPTION_RESNET_V2
#red = tf.keras.applications.inception_resnet_v2.InceptionResNetV2()
#tf.keras.applications.inception_resnet_v2.preprocess_input   //299

#RESNET101
#red = tf.keras.applications.resnet.ResNet101()
#tf.keras.applications.resnet.preprocess_input      //224

#NASNET_MOBILE
#red = tf.keras.applications.nasnet.NASNetMobile()
#tf.keras.applications.nasnet.preprocess_input      //224

#EFFICIENTNETB3
#red = tf.keras.applications.efficientnet.EfficientNetB3()
#tf.keras.applications.efficientnet.preprocess_input    //224

#INCEPTION_V3
#red = tf.keras.applications.inception_v3.InceptionV3()
#tf.keras.applications.inception_v3.preprocess_input    //299

#DENSENET201
#red = tf.keras.applications.densenet.DenseNet201()
#tf.keras.applications.densenet.preprocess_input      //224

#EFFICIENTNETB7
red = tf.keras.applications.efficientnet.EfficientNetB7()
#tf.keras.applications.efficientnet.preprocess_input    //224

#XCEPTION
#red = tf.keras.applications.xception.Xception()
#tf.keras.applications.xception.preprocess_input //299

#MOBILENET_V2
#red = tf.keras.applications.mobilenet_v2.MobileNetV2()          #224
#tf.keras.applications.mobilenet.preprocess_input   //224

#NASNETMOBILE
#red = tf.keras.applications.nasnet.NASNetMobile()               #224
#tf.keras.applications.nasnet.preprocess_input   //224

TRAINING_DIR = "/content/drive/My Drive/Dataset"
ss = ImageDataGenerator(preprocessing_function=tf.keras.applications.efficientnet.preprocess_input).flow_from_directory(
    TRAINING_DIR, target_size= (224,224), batch_size = 10)

test_data = "/content/drive/My Drive/Dataset"
test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.efficientnet.preprocess_input).flow_from_directory(
    test_data, target_size= (224,224), batch_size = 1,shuffle = False)

#red.summary()

x = red.layers[-2].output #se elimina las dos ulitmas capas y se agrega la capa con 20 etiquetas de salida
pred = tf.keras.layers.Dense(20, activation=tf.nn.softmax)(x)
modelo = tf.keras.Model(inputs = red.input, outputs = pred)

#modelo.summary()

for layer in modelo.layers[:-5]: #se entrena las ultimas 5 capas
    layer.trainable= False

modelo.compile(loss = 'categorical_crossentropy', optimizer = 'rmsprop', metrics = ['accuracy'])

history = modelo.fit(ss,epochs=10 , verbose= 1)

acc = history.history['accuracy']
loss = history.history['loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.title('Training accuracy')
plt.legend(loc=0)
plt.figure()

plt.plot(epochs, loss, 'g', label='Loss')
plt.title('Loss')
plt.legend(loc=0)
plt.figure()

plt.show()

classes = modelo.predict(test_batches)
predicted=np.argmax(classes,axis=1)
print(predicted)

import pandas as pd
filenames = test_batches.filenames
result=pd.DataFrame({"Filename":filenames,"Predictions":predicted})

#TARGET
real_class_indices=[]                     #Se obtiene el Target de cada imagen 
for i in range(0, len(filenames)):        #Por medio del nombre de cada imagen se la identifica y se le da el valor de su indice en el vector de predicciones 
  if ("_A_" in filenames[i]):               #Tenemos como resultado la etiqueta real, a quien pertenece la imagen para posteriarmente hacer una comparacion 
    real_class_indices.append(0)
  if ("_B_" in filenames[i]):
    real_class_indices.append(11)
  if ("_C_" in filenames[i]):
    real_class_indices.append(13)
  if ("_D_" in filenames[i]):
    real_class_indices.append(14)
  if ("_E_" in filenames[i]):
    real_class_indices.append(15)
  if ("_F_" in filenames[i]):
    real_class_indices.append(16)
  if ("_G_" in filenames[i]):
    real_class_indices.append(17)
  if ("_H_" in filenames[i]):
    real_class_indices.append(18)
  if ("_I_" in filenames[i]):
    real_class_indices.append(19)
  if ("_J_" in filenames[i]):
    real_class_indices.append(1)
  if ("_K_" in filenames[i]):
    real_class_indices.append(2)
  if ("_L_" in filenames[i]):
    real_class_indices.append(3)
  if ("_M_" in filenames[i]):
    real_class_indices.append(4)
  if ("_N_" in filenames[i]):
    real_class_indices.append(5)
  if ("_Z_" in filenames[i]):
    real_class_indices.append(6)
  if ("_O_" in filenames[i]):
    real_class_indices.append(7)
  if ("_P_" in filenames[i]):
    real_class_indices.append(8)
  if ("_Q_" in filenames[i]):
    real_class_indices.append(9)
  if ("_R_" in filenames[i]):
    real_class_indices.append(10)
  if ("_S_" in filenames[i]):
    real_class_indices.append(12)
print(real_class_indices)

#matriz de confusion
data = pd.DataFrame({"Predic":predicted,"Originales":real_class_indices})
matriz = pd.crosstab (data ['Predic'], data ['Originales'], rownames = ['Predicciones'], colnames = ['Originales'])
plt.figure(figsize = (10,7))
sn.heatmap (matriz, annot = True, fmt=" ", annot_kws={"size": 11})

data['col3'] = np.where(data['Predic'] == data['Originales'], 0, 1)
pd.options.display.max_rows = None
x = data.sum()
x[2]

#indices de rendimiento
real = np.array(real_class_indices)
print(classification_report(real_class_indices,predicted))

modelo.save('/content/drive/My Drive//EfficientB7') #se guarda los pesos del modelo entrenado
