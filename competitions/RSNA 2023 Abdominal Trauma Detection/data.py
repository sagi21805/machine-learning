import os
# You can use `tensorflow`, `pytorch`, `jax` here
# KerasCore makes the notebook backend agnostic :)
import time
import numpy as np
import pandas as pd
import tensorflow as tf
st = time.time()
class Config:
    SEED = 42
    IMAGE_SIZE = [256, 256]
    BATCH_SIZE = 10
    EPOCHS = 10
    TARGET_COLS  = [
        "patient_id",
        "bowel_injury", "extravasation_injury",
        "kidney_healthy", "kidney_low", "kidney_high",
        "liver_healthy", "liver_low", "liver_high",
        "spleen_healthy", "spleen_low", "spleen_high",
    ]
    AUTOTUNE = tf.data.AUTOTUNE
    TRAIN_LABEL_PATH = "/home/sagi21805/Desktop/Vscode/machine-learning/competitions/RSNA 2023 Abdominal Trauma Detection/train.csv"
    TRAIN_IMG_PATH = "/home/sagi21805/Desktop/Vscode/machine-learning/competitions/RSNA 2023 Abdominal Trauma Detection/rsna_256x256_jpeg"
    
    train_data = pd.read_csv(TRAIN_LABEL_PATH); train_data = train_data.drop_duplicates()

    # imgs_paths = [x for patient_id in os.listdir(TRAIN_IMG_PATH) ]/
    # get image_paths and labels
    print("[INFO] Building the dataset...")
    id_label_dict = {label[0]: label[1: ] for label in train_data[TARGET_COLS].values}
    img_paths = []
    labels = []
    for patient_id in os.listdir(TRAIN_IMG_PATH):
        for img_path in os.listdir(TRAIN_IMG_PATH + "/" + patient_id):
            img_paths.append(TRAIN_IMG_PATH + "/" + patient_id + "/" + img_path)
            labels.append(id_label_dict[int(img_paths[-1].split("/")[-2])])            
    



    def decode_image_and_label(self, img_path: str, id):
        file_bytes = tf.io.read_file(img_path)
        image = tf.io.decode_jpeg(file_bytes, channels=1)
        image = tf.cast(image, tf.float32) / 255.0
        label = tf.cast(id, tf.float32)
        #         bowel       fluid       kidney      liver       spleen
        labels = (label[0:1], label[1:2], label[2:5], label[5:8], label[8:11])
        return (image, labels)



    def build_dataset(self):
        ds = (
            tf.data.Dataset.from_tensor_slices((self.img_paths, self.labels))\
            .map(self.decode_image_and_label, num_parallel_calls=config.AUTOTUNE)\
            .shuffle(config.BATCH_SIZE * 10)\
            .batch(config.BATCH_SIZE)\
            .prefetch(config.AUTOTUNE)\
            .batch(config.BATCH_SIZE)
        )
        
        return ds

def build_model():
    # Define Input
    inputs = tf.keras.Input(shape=config.IMAGE_SIZE , batch_size=config.BATCH_SIZE)
    l1 = tf.keras.layers.Dense(units = 32, activation='silu')(inputs)
    print("[INFO] Building the model...")
    model = tf.keras.Model(inputs=inputs)
    optimizer = tf.keras.optimizers.Adam()
    loss = tf.keras.losses.CategoricalCrossentropy()
    metrics = "accuracy"
    print("[INFO] Compiling the model...")
    model.compile(
        optimizer=optimizer,
      loss=loss,
      metrics=metrics
    )
    
    return model

config = Config()
ds = config.build_dataset()


for d in ds:
    print(d)
    break
print(f"\n\n\ntime: {time.time() - st}")
print('DONE')

tf.keras.utils.set_random_seed(seed=config.SEED)



# train and valid dataset
# train_ds = build_dataset(labels=train_labels)

# # build the model
# print("[INFO] Building the model...")
# model = build_model()
# # train
# print("[INFO] Training...")
# history = model.fit(
#     train_ds,
#     epochs=config.EPOCHS,
# )


