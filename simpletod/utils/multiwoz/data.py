import logging
import shutil
import urllib
import json
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile
from typing import Dict, Tuple

DATA_DIR = Path('./resources/')


def _copy_files(src: Path, dst: Path) -> None:
    shutil.copy(str(src / 'data.json'), str(dst))
    shutil.copy(str(src / 'valListFile.json'), str(dst))
    shutil.copy(str(src / 'testListFile.json'), str(dst))
    shutil.copy(str(src / 'dialogue_acts.json'), str(dst))


def _load_json(path: Path) -> Dict:
    return json.loads(path.read_text())


def load_multiwoz() -> Tuple[Dict, Dict]:
    data_path = DATA_DIR / 'multi-woz-2.1/data.json'

    download_path = DATA_DIR / 'multi-woz/'
    download_path.mkdir(exist_ok=True, parents=True)

    extract_path = download_path / 'MULTIWOZ2.1/'

    if not data_path.exists():
        download_multiwoz(str(download_path))
        _copy_files(src=extract_path, dst=download_path)

    data = _load_json(DATA_DIR / 'multi-woz/data.json')
    dial_acts = _load_json(DATA_DIR / 'multi-woz/dialogue_acts.json')

    return data, dial_acts


def download_multiwoz(
        download_path: str,
        dataset_url: str = "https://www.repository.cam.ac.uk/bitstream/handle/1810/294507/MULTIWOZ2.1.zip?sequence=1&isAllowed=y", # noqa
) -> None:
    logging.info("Downloading and unzipping the MultiWOZ dataset")
    resp = urllib.request.urlopen(dataset_url)
    zip_ref = ZipFile(BytesIO(resp.read()))
    zip_ref.extractall(download_path)
    zip_ref.close()
