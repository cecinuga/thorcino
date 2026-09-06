from os import path
from pathlib import Path
import pickle
import numpy as np
from dataclasses import dataclass


@dataclass(frozen=True)
class Artifact:
    """The whole data of an artifact."""
    data: dict[str, np.ndarray]

def load_artifact(artifact: str|Path) -> Artifact:
    """Data of one artifact, pickle format."""
    folder = path.exists(artifact)

    with open(artifact, 'rb') as f:
        data = pickle.load(f)

    return Artifact(data)