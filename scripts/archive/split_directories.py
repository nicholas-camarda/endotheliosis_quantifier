import splitfolders

input_folder = "data/mitochondria_data/all_data"
output_folder = "data/mitochondria_data/data_batched"
splitfolders.ratio(input_folder, output_folder, seed=42,
                   ratio=(0.8, 0.2), group_prefix=None)
