import tensorflow as tf
import numpy as np
import pandas as pd 

#gameplan -->
    #for each id, pass all the images through the net,
    #for each img collect the result 
    #sum the result and devide by the total imgs of the patient
    #soft max the vals of the liver, spleen, and kidney 
    #sigmoid the values of the bowl and extravasation and 1 - healthy for the injury


config = {
    "SEED" : 42, 
    "IMAGE_SHAPE" : [256, 256] , 
    "BATCH_SIZE" : 1, 
    "EPOCHS" : 1,
    "TARGET_CLASSES"  : [
        
        "bowel_injury", #bowel_healthy is just 1 - the chance of the injury
        "extravasation_injury", #bextravasation_healthy is just 1 - the chance of the injury
        "kidney_healthy", "kidney_low", "kidney_high",
        "liver_healthy", "liver_low", "liver_high",
        "spleen_healthy", "spleen_low", "spleen_high",
        
    ],
    "AUTOTUNE" : tf.data.AUTOTUNE
}

def build_model():
    # Define Input
    inputs = tf.keras.Input(shape=config["IMAGE_SHAPE"] + [3, ], batch_size=config["BATCH_SIZE"])
    
    # Define Backbon
    # Define 'necks' for each head
    x_bowel = tf.keras.layers.Dense(32, activation='silu')(inputs)
    x_extra = tf.keras.layers.Dense(32, activation='silu')(inputs)
    x_kidney = tf.keras.layers.Dense(32, activation='silu')(inputs)
    x_liver = tf.keras.layers.Dense(32, activation='silu')(inputs)
    x_spleen =tf.keras.layers.Dense(32, activation='silu')(inputs)

    # Define heads
    out_bowel = tf.keras.layers.Dense(1, name='bowel', activation='sigmoid')(x_bowel) # use sigmoid to convert predictions to [0-1]
    out_extra = tf.keras.layers.Dense(1, name='extra', activation='sigmoid')(x_extra) # use sigmoid to convert predictions to [0-1]
    out_kidney = tf.keras.layers.Dense(3, name='kidney', activation='softmax')(x_kidney) # use softmax for the kidney head
    out_liver = tf.keras.layers.Dense(3, name='liver', activation='softmax')(x_liver) # use softmax for the liver head
    out_spleen = tf.keras.layers.Dense(3, name='spleen', activation='softmax')(x_spleen) # use softmax for the spleen head
    
    # Concatenate the outputs
    outputs = [out_bowel, out_extra,out_kidney, out_liver, out_spleen]

    # Create model
    print("[INFO] Building the model...")
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    

    # Compile the model
    optimizer = tf.keras.optimizers.Adam()
    loss = {
        "bowel":tf.keras.losses.BinaryCrossentropy(),
        "extra":tf.keras.losses.BinaryCrossentropy(),
        "kidney":tf.keras.losses.CategoricalCrossentropy(),
        "liver":tf.keras.losses.CategoricalCrossentropy(),
        "spleen":tf.keras.losses.CategoricalCrossentropy(),
    }
    metrics = {
        "bowel":["accuracy"],
        "extra":["accuracy"],
        "kidney":["accuracy"],
        "liver":["accuracy"],
        "spleen":["accuracy"],
    }
    print("[INFO] Compiling the model...")
    model.compile(
        optimizer=optimizer,
      loss=loss,
      metrics=metrics
    )
    return model

model = build_model()
def decode_image_and_label(label):
    image = tf.random.uniform(config["IMAGE_SHAPE"] + [3, ], 0, 255)
    image = tf.cast(image, tf.float32) / 255.0
    label = tf.cast(label, tf.float32)
    #         bowel       extra      kidney      liver       spleen
    labels = (label[0:1], label[1:2], label[2:5], label[5:8], label[8:11])
    return (image, labels)


dataframe = pd.read_csv(f"./train.csv")
x = 1
train_labels = dataframe[config["TARGET_CLASSES"]].values.astype(np.float32)[0: x]
train_ds = (tf.data.Dataset.from_tensor_slices((train_labels))\
.map(decode_image_and_label, num_parallel_calls=config["AUTOTUNE"])\
.batch(config["BATCH_SIZE"])\
.prefetch(config["AUTOTUNE"]))

print("****************** \n\n")
for t in train_ds:
    print(type(train_ds))
    print(type(t))
    for x in t:
        print(x)
        print("\n\n")
        print(type(x))
        print("\n\n")
    break
print("******************\n\n")

print(train_ds)

print("******************\n\n")

model.fit(train_ds, batch_size=config["BATCH_SIZE"])
