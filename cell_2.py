import pandas as pd
import shutil
import os
from tqdm import tqdm

df = pd.read_csv('data/HAM10000_metadata.csv')

class_map = {
    'mel': 'Melanoma', 'nv': 'Nevus', 'bcc': 'BCC',
    'akiec': 'AK', 'bkl': 'BKL', 'df': 'DF', 'vasc': 'VASC'
}

for cls in class_map.values():
    os.makedirs(f'data/{cls}', exist_ok=True)

image_dirs = ['data/HAM10000_images_part_1', 'data/HAM10000_images_part_2']

for _, row in tqdm(df.iterrows(), total=len(df)):
    img_id = row['image_id']
    dx = row['dx']
    cls_name = class_map.get(dx, 'Other')
    for d in image_dirs:
        src = f'{d}/{img_id}.jpg'
        if os.path.exists(src):
            shutil.copy(src, f'data/{cls_name}/{img_id}.jpg')
            break

print("✅ Dataset organized successfully!")
