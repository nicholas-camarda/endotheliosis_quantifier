import ast
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
TECH_DOCS_DIR = REPO_ROOT / ".agent-os" / "product" / "technical"

TARGET_DIRS = [
	REPO_ROOT / "scripts" / "main",
	REPO_ROOT / "scripts" / "utils",
	REPO_ROOT / "src",
]

@dataclass
class FunctionInfo:
	name: str
	lineno: int
	docstring: str
	args: List[str]
	calls: Set[str] = field(default_factory=set)

@dataclass
class ModuleAnalysis:
	path: Path
	functions: Dict[str, FunctionInfo] = field(default_factory=dict)
	imports: Set[str] = field(default_factory=set)


def discover_python_files(target_dirs: List[Path]) -> List[Path]:
	python_files: List[Path] = []
	for base in target_dirs:
		if not base.exists():
			continue
		for root, _, files in os.walk(base):
			for f in files:
				if f.endswith(".py"):
					python_files.append(Path(root) / f)
	return python_files


def analyze_module(file_path: Path) -> ModuleAnalysis:
	source = file_path.read_text(encoding="utf-8")
	tree = ast.parse(source)
	analysis = ModuleAnalysis(path=file_path)

	# Map of defined function names in this module for reference
	defined_functions: Set[str] = set()

	for node in ast.walk(tree):
		if isinstance(node, ast.Import):
			for alias in node.names:
				analysis.imports.add(alias.name)
		elif isinstance(node, ast.ImportFrom):
			if node.module:
				analysis.imports.add(node.module)
		elif isinstance(node, ast.FunctionDef):
			defined_functions.add(node.name)

	# Second pass: collect function details and calls
	for node in tree.body:
		if isinstance(node, ast.FunctionDef):
			docstring = ast.get_docstring(node) or ""
			args = [a.arg for a in node.args.args]
			func_info = FunctionInfo(name=node.name, lineno=node.lineno, docstring=docstring, args=args)

			# Collect simple call graph: calls to other functions (by simple name)
			calls: Set[str] = set()
			for sub in ast.walk(node):
				if isinstance(sub, ast.Call):
					# Only capture direct Name calls foo() not obj.foo()
					if isinstance(sub.func, ast.Name):
						calls.add(sub.func.id)
					elif isinstance(sub.func, ast.Attribute) and isinstance(sub.func.value, ast.Name):
						# Capture simple obj.method pattern name
						calls.add(sub.func.attr)
			func_info.calls = calls
			analysis.functions[node.name] = func_info

	return analysis


def render_function_inventory(analyses: List[ModuleAnalysis]) -> str:
	lines: List[str] = []
	all_functions: List[Tuple[Path, FunctionInfo]] = []
	for mod in analyses:
		for f in mod.functions.values():
			all_functions.append((mod.path, f))

	lines.append("# Function Inventory")
	lines.append("")
	lines.append(f"Total functions: {len(all_functions)}")
	lines.append("")

	# Group by file
	by_file: Dict[Path, List[FunctionInfo]] = {}
	for path, finfo in all_functions:
		by_file.setdefault(path, []).append(finfo)

	for path in sorted(by_file.keys()):
		rel = path.relative_to(REPO_ROOT)
		lines.append(f"## {rel}")
		for finfo in sorted(by_file[path], key=lambda x: x.lineno):
			short_doc = (finfo.docstring.strip().splitlines()[0] if finfo.docstring else "").strip()
			arg_list = ", ".join(finfo.args)
			lines.append(f"- {finfo.name}({arg_list})  — line {finfo.lineno}{' — ' + short_doc if short_doc else ''}")
		lines.append("")

	return "\n".join(lines)


def render_dependency_analysis(analyses: List[ModuleAnalysis]) -> str:
	lines: List[str] = []
	lines.append("# Function Dependency Analysis")
	lines.append("")

	for mod in sorted(analyses, key=lambda m: str(m.path)):
		rel = mod.path.relative_to(REPO_ROOT)
		lines.append(f"## {rel}")
		if mod.imports:
			imports_sorted = sorted(mod.imports)
			lines.append(f"- Imports: {', '.join(imports_sorted)}")
		else:
			lines.append("- Imports: (none)")

		if mod.functions:
			lines.append("- Function calls (intra-module and simple attribute calls):")
			for fname in sorted(mod.functions.keys()):
				finfo = mod.functions[fname]
				if finfo.calls:
					calls_sorted = ", ".join(sorted(finfo.calls))
					lines.append(f"  - {fname} → {calls_sorted}")
				else:
					lines.append(f"  - {fname} → (no calls detected)")
		else:
			lines.append("- Functions: (none)")
		lines.append("")

	return "\n".join(lines)


def main() -> None:
	TECH_DOCS_DIR.mkdir(parents=True, exist_ok=True)
	files = discover_python_files(TARGET_DIRS)
	analyses = [analyze_module(p) for p in files]

	inventory_md = render_function_inventory(analyses)
	(REPO_ROOT / ".agent-os" / "product" / "technical" / "function_inventory.md").write_text(inventory_md, encoding="utf-8")

	deps_md = render_dependency_analysis(analyses)
	(REPO_ROOT / ".agent-os" / "product" / "technical" / "function_dependency_analysis.md").write_text(deps_md, encoding="utf-8")

	print("Wrote function inventory and dependency analysis to .agent-os/product/technical/")


if __name__ == "__main__":
	main()
