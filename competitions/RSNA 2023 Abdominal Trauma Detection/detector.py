import os
import platform
import time
import numpy as np
import pandas as pd
import tensorflow as tf
st = time.time()

config = {
    'SEED' : 42,
    'IMAGE_SIZE' : [256, 256],
    'BATCH_SIZE' : 10,
    'EPOCHS' : 10,
    'TARGET_CLASSES'  : [
        "patient_id",
        "bowel_injury", "extravasation_injury",
        "kidney_healthy", "kidney_low", "kidney_high",
        "liver_healthy", "liver_low", "liver_high",
        "spleen_healthy", "spleen_low", "spleen_high",
    ],

    'AUTOTUNE' : tf.data.AUTOTUNE,
    
    'TRAIN_LABEL_PATH' : "/home/sagi21805/Desktop/Vscode/machine-learning/competitions/RSNA 2023 Abdominal Trauma Detection/train.csv",
    'TRAIN_LABEL_PATH_WINDOWS' : r"C:\VsCode\python\machineLearning\machine-learning\competitions\RSNA 2023 Abdominal Trauma Detection\train.csv",
    'TRAIN_IMG_PATH' : "/home/sagi21805/Desktop/Vscode/machine-learning/competitions/RSNA 2023 Abdominal Trauma Detection/rsna_256x256_jpeg",
    'TRAIN_IMG_PATH_WINDOWS' : r"C:\VsCode\python\machineLearning\machine-learning\competitions\RSNA 2023 Abdominal Trauma Detection\rsna_256x256_jpeg"
}

TRAIN_LABEL_PATH = config["TRAIN_LABEL_PATH_WINDOWS"] if platform.system() == "Windows" else config["TRAIN_LABEL_PATH"]
TRAIN_IMG_PATH = config["TRAIN_IMG_PATH_WINDOWS"] if platform.system() == "Windows" else config["TRAIN_IMG_PATH"]

train_data = pd.read_csv(TRAIN_LABEL_PATH); train_data = train_data.drop_duplicates()


def decode_image_and_label(img_path: str, label):
    file_bytes = tf.io.read_file(img_path)
    image = tf.io.decode_jpeg(file_bytes)
    image = tf.reshape(image, config["IMAGE_SIZE"])
    image = tf.cast(image, tf.float32) / 255.0
    label = tf.cast(label, tf.float32)
    #         bowel       fluid       kidney      liver       spleen
    labels = (label[0:1], label[1:2], label[2:5], label[5:8], label[8:11])
    return (image, labels)

def build_dataset(): 
    id_label_dict = {label[0]: label[1: ] for label in train_data[config["TARGET_CLASSES"]].values}
    img_paths = []
    labels = []
    slash = "\\" if platform.system() == "Windows" else "/" 
    for patient_id in os.listdir(TRAIN_IMG_PATH):
        for img_path in os.listdir(TRAIN_IMG_PATH + slash + patient_id):
            img_paths.append(TRAIN_IMG_PATH + slash + patient_id + slash + img_path)
            labels.append(id_label_dict[int(img_paths[-1].split(slash)[-2])]) 

    if len(img_paths) != len(labels):
        raise Exception("\n***************\n\
                        img_paths and labels must be in the same length\
                        \n***************\n")  

    ds = tf.data.Dataset.from_tensor_slices((img_paths, labels))\
        .map(decode_image_and_label, num_parallel_calls=config["AUTOTUNE"])\
        .shuffle(config["BATCH_SIZE"] * 10)\
        .batch(config["BATCH_SIZE"])\
        .prefetch(config["AUTOTUNE"])\
        .unbatch()\
        .batch(config["BATCH_SIZE"])
    
    
    return ds

def build_model():
    inputs = tf.keras.Input(shape=config["IMAGE_SIZE"], batch_size=config["BATCH_SIZE"])

    #*bowel 

    bowel1 = tf.keras.layers.Dense(units = 32, activation = tf.keras.activations.selu)(inputs)
    bowel_out = tf.keras.layers.Dense(name = "bowel", units = 1, activation = tf.keras.activations.sigmoid)(bowel1)
    
    #*extravasation 

    extra1 = tf.keras.layers.Dense(units = 32, activation = tf.keras.activations.selu)(inputs)
    extra_out = tf.keras.layers.Dense(name = "extra", units = 1, activation = tf.keras.activations.sigmoid)(extra1)  
    
    #*kidney 

    kidney1 = tf.keras.layers.Dense(units = 32, activation = tf.keras.activations.selu)(inputs)
    kidney_out = tf.keras.layers.Dense(name = "kidney", units = 3, activation = tf.keras.activations.softmax)(kidney1)
    
    #*liver 

    liver1 = tf.keras.layers.Dense(units = 32, activation = tf.keras.activations.selu)(inputs)
    liver_out = tf.keras.layers.Dense(name = "liver", units = 3, activation = tf.keras.activations.softmax)(liver1)
    
    #*spleen 

    spleen1 = tf.keras.layers.Dense(units = 32, activation = tf.keras.activations.selu)(inputs)
    spleen_out = tf.keras.layers.Dense(name = "spleen", units = 3, activation = tf.keras.activations.softmax)(spleen1)

    outputs = [bowel_out, extra_out, kidney_out, liver_out, spleen_out]

    #*compile config
    optimizer = tf.keras.optimizers.Adam()

    loss = {
        "bowel":tf.keras.losses.BinaryCrossentropy(from_logits=True),
        "extra":tf.keras.losses.BinaryCrossentropy(from_logits=True),
        "liver":tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        "kidney":tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        "spleen":tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    }

    metrics = {
        "bowel":["accuracy"],
        "extra":["accuracy"],
        "liver":["accuracy"],
        "kidney":["accuracy"],
        "spleen":["accuracy"],
    }

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model


print("[INFO]: building model")
st = time.time()
model = build_model()
print(f"DONE\n[TIME]: {time.time() - st}")
print("[INFO]: building dataframe")
st = time.time()
data = build_dataset()
print(f"DONE\n[TIME]: {time.time() - st}")


print("[INFO]: started Training")
# model.fit(data, batch_size = config["BATCH_SIZE"], epochs = config["EPOCHS"])

    


