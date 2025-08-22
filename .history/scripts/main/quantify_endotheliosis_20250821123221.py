
import os

def main() -> None:
	"""Delegate execution to eq.pipeline.quantify_endotheliosis to keep a single entrypoint."""
	from eq.pipeline.quantify_endotheliosis import run_random_forest, load_pickled_data
	
	# Load data and run analysis
	top_data_directory = 'data/preeclampsia_data'
	regression_cache_dir_path = os.path.join(top_data_directory, 'cache', 'features_and_scores')
	top_output_directory_regresion_models = 'output/regression_models'
	directory_rf_model = os.path.join(top_output_directory_regresion_models, 'rf-model')
	
	# Load up the glomeruli features from VGG16 and 0-3 grades for each of the images
	X_train = load_pickled_data(os.path.join(regression_cache_dir_path, 'X_train_glom_features.pkl'))
	y_train = load_pickled_data(os.path.join(regression_cache_dir_path, 'y_train_scores.pkl'))
	X_val = load_pickled_data(os.path.join(regression_cache_dir_path, 'X_val_glom_features.pkl'))
	y_val = load_pickled_data(os.path.join(regression_cache_dir_path, 'y_val_scores.pkl'))
	
	print(f'X_train: {X_train.shape}')
	print(f'y_train: {y_train.shape}')
	print(f'X_val: {X_val.shape}')
	print(f'y_val: {y_val.shape}')
	
	# Run random forest analysis
	run_random_forest(X_train, y_train, X_val, y_val,
					  model_output_directory=directory_rf_model,
					  n_estimators=300)

if __name__ == "__main__":
	main()
