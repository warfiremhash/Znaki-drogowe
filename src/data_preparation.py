import os
from sklearn.model_selection import train_test_split
import shutil

input_path = 'dataset/raw'
output_path = 'dataset/split'

train_path = os.path.join(output_path, 'train')
val_path = os.path.join(output_path, 'val')
test_path = os.path.join(output_path, 'test')

os.makedirs(train_path, exist_ok=True)
os.makedirs(val_path, exist_ok=True)
os.makedirs(test_path, exist_ok=True)


def split_data(input_path, train_path, val_path, test_path, train_ratio=0.7, val_ratio=0.2):
    for class_folder in os.listdir(input_path):
        class_folder_path = os.path.join(input_path, class_folder)
        if os.path.isdir(class_folder_path):
            images = os.listdir(class_folder_path)

            train, temp = train_test_split(images, train_size=train_ratio, random_state=42)
            val, test = train_test_split(temp, test_size=0.5, random_state=42)

            for subset, subset_path in zip([train, val, test], [train_path, val_path, test_path]):
                output_class_folder = os.path.join(subset_path, class_folder)
                os.makedirs(output_class_folder, exist_ok=True)
                for img_name in subset:
                    shutil.copy(os.path.join(class_folder_path, img_name), os.path.join(output_class_folder, img_name))


split_data(input_path, train_path, val_path, test_path)
print("Dane podzielone na zbiory: train, val, test.")
