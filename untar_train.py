import glob
import os
import shutil
import sys
import tarfile
from pathlib import Path

valohai_input_path = Path(sys.argv[1]).parent
archives = glob.glob(str(valohai_input_path / '*.tar.gz'))

# untar archives
for archive in archives:
    tar = tarfile.open(valohai_input_path / archive, "r:gz")
    tar.extractall(valohai_input_path)
    tar.close()

correct_jsons = glob.glob(str(valohai_input_path / '*.json'))

if (len(correct_jsons) > 0):
    # delete jsons from archive
    active_json_files = glob.glob(
        str(valohai_input_path / 'dataset' / '*.json'))
    for json in active_json_files:
        (valohai_input_path / 'dataset' / json).unlink()

    for json in correct_jsons:
        source_json = valohai_input_path / json
        dest_json = valohai_input_path / 'dataset'
        shutil.copy(str(source_json), str(dest_json))