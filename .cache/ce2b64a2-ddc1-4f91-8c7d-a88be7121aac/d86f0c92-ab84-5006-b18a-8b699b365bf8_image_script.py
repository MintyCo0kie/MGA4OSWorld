from PIL import Image
import hashlib
import os

pictures_dir = '/home/user/Pictures'
pictures = os.listdir(pictures_dir)

def calculate_image_hash(image_path):
    with Image.open(image_path) as img:
        img_byte_arr = img.tobytes()
        hash_result = hashlib.sha256(img_byte_arr).hexdigest()
    return hash_result

original_hash_map = {}
for picture in pictures:
    image_path = os.path.join(pictures_dir, picture)
    image_hash = calculate_image_hash(image_path)
    original_hash_map[image_hash] = picture
print(original_hash_map)