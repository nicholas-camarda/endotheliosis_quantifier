import os
import runpy


def main() -> None:
	"""Delegate execution to eq.pipeline.quantify_endotheliosis to keep a single entrypoint."""
	from eq.pipeline.quantify_endotheliosis import run_random_forest, load_pickled_data
	# Import and run the main functionality
	# This is a placeholder - the actual execution logic would be implemented here
	print("eq.pipeline.quantify_endotheliosis imported successfully")

if __name__ == "__main__":
	main()
