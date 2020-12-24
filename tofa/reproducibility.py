import subprocess
import sys
import tarfile
from logging import getLogger

import numpy as np
import torch

from tofa.filesystem import as_path

log = getLogger(__name__)


def set_all_random_seeds(seed):
    torch.random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def configure_reproducibility(seed=None, deterministic=False):
    if seed is not None:
        set_all_random_seeds(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def get_git_revision():
    try:
        rev = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
        branch = (
            subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"])
            .decode()
            .strip()
        )
        return branch, rev
    except subprocess.CalledProcessError:
        return None, None


def get_versions_dict():
    return {
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "troch": torch.__version__,
        "torhc.rev": torch.version.git_version,
    }


def create_code_snapshot(root, dst_path, extensions=(".py", ".json")):
    """Creates tarball with the source code"""
    log.info("Crate tar ball at: {}".format(root))
    with tarfile.open(str(dst_path), "w:gz") as tar:
        for path in as_path(root).rglob("*"):
            if ".git" in path.parts:
                continue
            if path.suffix.lower() in extensions:
                tar.add(
                    path.as_posix(),
                    arcname=path.relative_to(root).as_posix(),
                    recursive=True,
                )


class TeedStream(object):
    """Copy stdout to the file"""

    def __init__(self, fname, mode="w"):
        self.file = open(str(fname), mode)
        self.stdout = sys.stdout
        sys.stdout = self

    def __del__(self):
        sys.stdout = self.stdout
        self.file.close()

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

    def flush(self):
        self.file.flush()
