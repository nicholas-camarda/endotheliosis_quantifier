from eq.io.convert_files_to_jpg import convert_tif_to_jpg

if __name__ == '__main__':
	import sys
	if len(sys.argv) != 3:
		print('Usage: python convert_tif_to_jpg.py <input_directory> <output_directory>')
		sys.exit(1)
	convert_tif_to_jpg(sys.argv[1], sys.argv[2])
