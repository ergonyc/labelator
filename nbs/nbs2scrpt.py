
from nbconvert import PythonExporter
import codecs
import nbformat
from pathlib import Path

notebook_path = Path.home() / "projects/SingleCell/Labelator/nbs"

export_path = 'Path.home() / "projects/SingleCell/Labelator/scripts"

if not export_path.exists():
    export_path.mkdir()

lbl8r_nbs = notebook_path.glob('lbl8r*.ipynb')

e2e_nbs = notebook_path.glob('e2e*.ipynb')

for nb in lbl8r_nbs:
    # Load your notebook
    with open(nb.as_posix()) as fh:
        notebook_node = nbformat.read(fh, as_version=4)

    # Convert to Python script
    python_exporter = PythonExporter()
    python_script, _ = python_exporter.from_notebook_node(notebook_node)

    # Write script file
    with codecs.open(export_path, 'w', encoding='utf-8') as fh:
        fh.write(python_script)


for nb in e2e_nbs:
    # Load your notebook
    with open(nb.as_posix()) as fh:
        notebook_node = nbformat.read(fh, as_version=4)

    # Convert to Python script
    python_exporter = PythonExporter()
    python_script, _ = python_exporter.from_notebook_node(notebook_node)

    # Write script file
    with codecs.open(export_path, 'w', encoding='utf-8') as fh:
        fh.write(python_script)