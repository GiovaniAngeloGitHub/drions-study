import os
import cv2
import numpy as np
from scipy.interpolate import splprep, splev

def generate_mask(image_path, annotation_path, output_path, img_size=(256, 256)):
    image = cv2.imread(image_path)
    height, width = image.shape[:2]

    points = np.loadtxt(annotation_path, delimiter=',')
    x, y = points[:, 0], points[:, 1]

    tck, u = splprep([x, y], s=0, per=True)
    u_new = np.linspace(u.min(), u.max(), 1000)
    x_new, y_new = splev(u_new, tck, der=0)
    contour = np.array([np.array([x_new, y_new]).T], dtype=np.int32)

    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, contour, 255)

    mask_resized = cv2.resize(mask, img_size)

    cv2.imwrite(output_path, mask_resized)

def generate_all_masks(image_dir, annotation_dir, output_dir, img_size=(256, 256)):
    os.makedirs(output_dir, exist_ok=True)
    image_files = sorted(os.listdir(image_dir))
    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        image_id = os.path.splitext(image_file)[0].split('_')[-1]
        annotation_file = f'anotExpert1_{image_id}.txt'
        annotation_path = os.path.join(annotation_dir, annotation_file)
        output_path = os.path.join(output_dir, f'image_{image_id}.png')
        if os.path.exists(annotation_path):
            generate_mask(image_path, annotation_path, output_path, img_size)
        else:
            print(f'Anotação não encontrada para {image_file}')
