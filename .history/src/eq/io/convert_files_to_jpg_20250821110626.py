import os
import sys
from PIL import Image


# script will now search for .tif files recursively in the input directory, 
# create the same directory structure in the output directory, and save 
# the converted .jpg files in their respective subdirectories.

def convert_tif_to_jpg(input_dir, output_dir):
    """Converts .tif files in input_dir into .jpg 
    and stores them in output_dir 

    Args:
        input_dir (str): _description_
        output_dir (str): _description_
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith('.tif'):
                input_file = os.path.join(root, file)
                relative_path = os.path.relpath(root, input_dir)
                output_subdir = os.path.join(output_dir, relative_path)

                if not os.path.exists(output_subdir):
                    os.makedirs(output_subdir)

                file_name_without_ext, _ = os.path.splitext(file)
                closest_parent_dir_name = os.path.basename(os.path.normpath(output_subdir))
                output_file_name = f"{closest_parent_dir_name}_{file_name_without_ext}.jpg"
                output_file = os.path.join(output_subdir, output_file_name)

                try:
                    img = Image.open(input_file)
                    img.convert('RGB').save(output_file, 'JPEG')
                except Exception as e:
                    print(f"Error converting {input_file}: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_tif_to_jpg.py <input_directory> <output_directory>")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    convert_tif_to_jpg(input_dir, output_dir)
