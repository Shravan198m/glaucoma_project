import nbformat
import sys
import os
from pathlib import Path

# Ensure src is in path
project_root = Path.cwd().resolve()
src_path = project_root.parent / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))
else:
    sys.path.insert(0, str(project_root.parent))

# Load notebook
notebook_path = Path("glaucoma_project_consolidated.ipynb")
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

# Execute each code cell
for i, cell in enumerate(nb.cells):
    if cell.cell_type == 'code':
        print(f"\n=== Executing cell {i} (id: {cell.id}) ===")
        # Print source for reference
        # print(cell.source)
        try:
            exec(cell.source, globals())
        except Exception as e:
            print(f"\nError in cell {i}: {e}")
            import traceback
            traceback.print_exc()
            # Optionally break
            break
else:
    print("\nAll cells executed successfully.")