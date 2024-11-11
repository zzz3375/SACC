import cv2
import numpy as np
import os

def yolo_to_mask(image_path, annotation_path, output_path, image_size):
    # Load image
    image = cv2.imread(image_path)
    height, width, _ = image.shape

    # Create a blank mask
    mask = np.zeros((height, width), dtype=np.uint8)

    # Read YOLO annotation file
    with open(annotation_path, 'r') as file:
        lines = file.readlines()

    for line in lines:
        # Parse YOLO annotation
        class_id =  line.strip().split()[0]
        coordinates = np.array(line.strip().split(), dtype=float)[1:].reshape((-1,2))
        # Convert YOLO format to pixel values
        coordinates[:,0] *= width
        coordinates[:,1] *= height
        coordinates = coordinates.astype(int)
        # Calculate bounding box coordinates

        # Draw the rectangle on the mask
        cv2.fillPoly(mask, [coordinates], 255)

    # Save the mask
    mask_filename = os.path.join(output_path, os.path.basename(image_path).replace('.jpg', '.png'))
    cv2.imwrite(mask_filename, mask)

if __name__ == '__main__':
    image_dir = r"C:\Users\13694\SACC\data\roboflow\train\images"
    annotation_dir = r"C:\Users\13694\SACC\data\roboflow\train\labels"
    output_dir = r"C:\Users\13694\SACC\data\roboflow\train\masks"

    os.makedirs(output_dir, exist_ok=True)

    for image_file in os.listdir(image_dir):
        if image_file.endswith('.jpg'):
            image_path = os.path.join(image_dir, image_file)
            annotation_path = os.path.join(annotation_dir, image_file.replace('.jpg', '.txt'))
            yolo_to_mask(image_path, annotation_path, output_dir, (416, 416))
    pass

# Example usage
