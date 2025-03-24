import os
import shutil
import subprocess
from tempfile import gettempdir
from pathlib import Path
from typing import List, Tuple, Dict, Any, Union


def chimera_env_variable():
    os.environ["PATH"] = "/home/hitesit/.local/UCSF-Chimera64-1.16/bin/:" + os.environ.get("PATH", "")


def write_chimera_script():
    dockprep_str = f"""
import chimera
import sys
from DockPrep import prep
models = chimera.openModels.list(modelTypes=[chimera.Molecule])
prep(models)
from WriteMol2 import writeMol2
writeMol2(models, "rec_prep.mol2")
    """

    chimera_py_path = os.path.join(gettempdir(), "chimera.py")
    with open(chimera_py_path, "w") as f:
        f.write(dockprep_str)

    return chimera_py_path


def run_chimera_script(pdb_file: Path, pdb_mol2: Path) -> Path:
    # Load the variable
    chimera_env_variable()

    # Write the script
    chimera_py_path = write_chimera_script()

    # Execute the script
    chimera_command = f"chimera --nogui {str(pdb_file)} {chimera_py_path}"
    subprocess.run(chimera_command, check=True, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # IO
    output_system = Path(os.path.join(gettempdir()), "rec_prep.mol2")
    shutil.move(output_system, pdb_mol2)

    return pdb_mol2
