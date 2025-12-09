import os
from PIL import Image
from pillow_heif import register_heif_opener

# Enable HEIC reading support
register_heif_opener()

# Paths
project_root = os.path.dirname(os.path.abspath(__file__))
images_folder = os.path.join(project_root, "data", "images")
output_folder = os.path.join(images_folder, "images_jpg")

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Convert all HEIC images
for filename in os.listdir(images_folder):
    if filename.lower().endswith(".heic"):
        heic_path = os.path.join(images_folder, filename)
        jpg_name = filename.rsplit(".", 1)[0] + ".jpg"
        jpg_path = os.path.join(output_folder, jpg_name)

        img = Image.open(heic_path)
        img.save(jpg_path, "JPEG", quality=95)

        print(f"Converted: {filename} → {jpg_name}")

print("\nConversion complete! All JPGs saved in images_jpg/")
