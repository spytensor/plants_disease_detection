import json
import shutil
import os
from glob import glob
from tqdm import tqdm

try:
    for i in range(0,59):
        os.mkdir("./data/train/" + str(i))
except:
    pass
    
file_train = json.load(open("./data/temp/labels/AgriculturalDisease_train_annotations.json","r",encoding="utf-8"))
file_val = json.load(open("./data/temp/labels/AgriculturalDisease_validation_annotations.json","r",encoding="utf-8"))

file_list = file_train + file_val

for file in tqdm(file_list):
    filename = file["image_id"]
    origin_path = "./data/temp/images/" + filename
    ids = file["disease_class"]
    if ids ==  44:
        continue
    if ids == 45:
        continue
    if ids > 45:
        ids = ids -2
    save_path = "./data/train/" + str(ids) + "/"
    shutil.copy(origin_path,save_path)

