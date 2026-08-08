# ruff: noqa
from __future__ import annotations
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from dotenv import load_dotenv
from PIL import Image, UnidentifiedImageError

DATASET_CONFIGURE = {
    "small_dataset": {
        "id": "small-dataset",
        "worksheet": "breast-usg.xlsx",
        "required_columns": [
            "CaseID",
            "Image_filename",
            "Mask_tumor_filename",
            "Mask_other_filename",
            "Classification",
        ],
    },
    "medium_dataset": {"id": "medium-dataset", "worksheet": "medium-dataset-780"},
    "big_dataset": {"id": "big-dataset", "worksheet": "big-dataset-9.248"},
}
CLASS_TO_INDEX = {
    "benign": 0,
    "malignant": 1,
}
