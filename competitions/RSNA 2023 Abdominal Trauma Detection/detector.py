import tensorflow as tf
import numpy as np
import pandas as pd 
import pydicom as dicom
import os 
import time
import matplotlib.pyplot as plt

#gameplan -->
    #for each id, pass all the images through the net,
    #for each img collect the result 
    #sum the result and devide by the total imgs of the patient
    #soft max the vals of the liver, spleen, and kidney 
    #sigmoid the values of the bowl and extravasation and 1 - healthy for the injury


config = {
    "SEED" : 42, 
    "IMAGE_SIZE" : [256, 256], 
    "BATCH_SIZE" : 64, 
    "EPOCHS" : 10,
    "TARGET_CLASSES"  : [
        
        "bowel_injury", #bowel_healthy is just 1 - the chance of the injury
        "extravasation_injury", #bextravasation_healthy is just 1 - the chance of the injury
        "kidney_healthy", "kidney_low", "kidney_high",
        "liver_healthy", "liver_low", "liver_high",
        "spleen_healthy", "spleen_low", "spleen_high",
        
    ]
}


BASE_PATH = "/kaggle/input/rsna-2023-abdominal-trauma-detection/"
st = time.time()
train_labels = pd.read_csv(f"{BASE_PATH}/train.csv")
patients_dict = train_labels.set_index('patient_id').T.to_dict('list')
images_list = os.listdir(f"{BASE_PATH}/train_images")
images_list.sort()

#*****CURRENTLY THE SERIES ID SORT DOESN'T MATTER**********
print("started")
for patience_id in images_list: 
    print(1)
    patience_images = []
    for series_id in os.listdir(f"{BASE_PATH}/train_images/{patience_id}"):
        for img_path in os.listdir(f"{BASE_PATH}/train_images/{patience_id}/{series_id}"):
            patience_images.append(np.array(dicom.dcmread(f"{BASE_PATH}/train_images/{patience_id}/{series_id}/{img_path}").pixel_array))
            print(patience_images[0].shape)
            plt.imshow(patience_images[0])
            plt.show()
            break
    patients_dict[patience_id] = (patience_images, patients_dict[patience_id])
    
print(patients_dict)
print(f"time: {time.time() - st}")

