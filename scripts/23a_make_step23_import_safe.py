from pathlib import Path

path = Path("scripts/23_build_pytorch_lazy_datasets.py")

text = path.read_text(encoding="utf-8")
lines = text.splitlines()

marker = 'section("CONSTRUCTING LAZY DATASETS")'

try:
    idx = next(i for i, line in enumerate(lines) if line.strip() == marker)
except StopIteration:
    raise RuntimeError(
        'Could not find section("CONSTRUCTING LAZY DATASETS")'
    )

# Do not modify twice.
if any(
    line.strip() == 'if __name__ == "__main__":'
    for line in lines[idx:]
):
    raise RuntimeError(
        "Step 23 appears to have already been made import-safe."
    )

before = lines[:idx]
executable = lines[idx:]

new_lines = []

new_lines.extend(before)

new_lines.append("")
new_lines.append("")
new_lines.append("def main():")
new_lines.extend(
    "    " + line if line.strip() else ""
    for line in executable
)

new_lines.append("")
new_lines.append("")
new_lines.append('if __name__ == "__main__":')
new_lines.append("    main()")
new_lines.append("")

backup = path.with_suffix(".py.before_import_safe.bak")

if backup.exists():
    raise RuntimeError(
        f"Backup already exists: {backup}"
    )

backup.write_text(text, encoding="utf-8")
path.write_text("\n".join(new_lines), encoding="utf-8")

print("=" * 90)
print("STEP 23 IMPORT-SAFETY REFACTOR")
print("=" * 90)
print()
print("Original:")
print(path)
print()
print("Backup:")
print(backup)
print()
print(f"Executable block started at original line: {idx + 1}")
print()
print("Refactor complete.")
print()
print("Reusable classes/functions remain module-level.")
print("Step-23 construction/audit now executes only through main().")
print()
print("No data files were modified.")