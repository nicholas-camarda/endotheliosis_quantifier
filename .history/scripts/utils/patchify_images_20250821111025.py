from eq.patches.patchify_images import patchify_image_dir

if __name__ == '__main__':
	import sys
	if len(sys.argv) != 4:
		print('Usage: python patchify_images.py <square_size> <input_dir> <output_dir>')
		sys.exit(1)
	square_size = int(sys.argv[1])
	input_dir = sys.argv[2]
	output_dir = sys.argv[3]
	patchify_image_dir(square_size, input_dir, output_dir)
