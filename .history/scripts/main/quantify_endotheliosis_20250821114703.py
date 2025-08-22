import os
import runpy


def main() -> None:
	"""Delegate execution to scripts/main/4_quantify_endotheliosis.py to keep a single entrypoint."""
	target_path = os.path.join(os.path.dirname(__file__), "4_quantify_endotheliosis.py")
	if not os.path.exists(target_path):
		raise FileNotFoundError(f"Expected target script not found: {target_path}")
	runpy.run_path(target_path, run_name="__main__")

if __name__ == "__main__":
	main()
