# -*- coding: utf-8 -*-
"""buildAndTrainBirdsVsSquirrels.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1WsWaGejnTznc1rsNTuSox4Hkmh2Jo8yM
"""

#connecting my notebook to my google drive
from google.colab import drive
drive.mount('/content/drive')
!ls "/content/drive/My Drive"

#loading in the path for tfrecords datasets
birdsVsSquirrelsTrainFilepath= '/content/drive/MyDrive/birds-vs-squirrels-train.tfrecords'
birdsVsSquirrelsValidationFilepath= '/content/drive/MyDrive/birds-vs-squirrels-validation.tfrecords'

#saving the tfrecords as datasets
import tensorflow as tf
rawBirdVsSquirrelTrain=tf.data.TFRecordDataset(birdsVsSquirrelsTrainFilepath)
rawBirdVsSquirrelValidation=tf.data.TFRecordDataset(birdsVsSquirrelsValidationFilepath)

#getting the number of samples in the training set for splitting the data 
datLen=rawBirdVsSquirrelTrain.reduce(0,lambda x,y: x+1)

#splitting the training and testing data
numTrain=int(datLen.numpy()*.9)
numTest=int(datLen.numpy()*.1)

#parsing the data so that we can get image/targets
feature_description={'image':tf.io.FixedLenFeature([],tf.string),'label':tf.io.FixedLenFeature([],tf.int64)}
def parse_examples(serialized_examples):
    examples=tf.io.parse_example(serialized_examples,feature_description)
    targets=examples.pop('label')
    images=tf.image.resize_with_pad(tf.cast(tf.io.decode_jpeg(
        examples['image'],channels=3),tf.float32),299,299)
    return images, targets

#calling parsing method on data
datasetTrain = rawBirdVsSquirrelTrain.take(numTrain).map(parse_examples, num_parallel_calls=tf.data.experimental.AUTOTUNE)
datasetValidation = rawBirdVsSquirrelValidation.map(parse_examples, num_parallel_calls=tf.data.experimental.AUTOTUNE)
datasetTest = rawBirdVsSquirrelTrain.skip(numTrain).take(numTest).map(parse_examples, num_parallel_calls=tf.data.experimental.AUTOTUNE)

#getting preprocess method from xception
def preproc(image,label):
    inp=tf.keras.applications.xception.preprocess_input(image)
    return inp,label

#calling preprocess method on data
datasetTrain_batched= datasetTrain.map(preproc,num_parallel_calls=tf.data.AUTOTUNE).batch(64)
datasetTest_batched =  datasetTest.map(preproc,num_parallel_calls=tf.data.AUTOTUNE).batch(64)
datasetValidation_batched =  datasetValidation.map(preproc,num_parallel_calls=tf.data.AUTOTUNE).batch(64)

#getting xception model in from website
base_model=tf.keras.applications.xception.Xception(weights='imagenet',include_top=False)

#creating our model
avg=tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
output=tf.keras.layers.Dense(3,activation="softmax")(avg)
model=tf.keras.models.Model(inputs=base_model.input,outputs=output)

for layer in base_model.layers:
    layer.trainable=False

#creating checkpoint in case model crashes
checkpoint_cb=tf.keras.callbacks.ModelCheckpoint('birdModTopFit.h5',save_best_only=True)
#creating early stopping
earlyStop_cb=tf.keras.callbacks.EarlyStopping(patience=10,restore_best_weights=True)
#creating step size
ss=.001
#using gradient descent 
optimizer=tf.keras.optimizers.SGD(learning_rate=ss)
#compiling the model
model.compile(loss="sparse_categorical_crossentropy",optimizer=optimizer,metrics=["accuracy"])

#fitting the model
model.fit(datasetTrain_batched,validation_data=datasetValidation_batched,epochs=10,callbacks=[checkpoint_cb,earlyStop_cb])

#evaluating the model on our testing set
model.evaluate(datasetTest_batched)

#saving the model 
tf.saved_model.save(model, '/content/drive/MyDrive/birdsVsSquirrelsModel')