"""Path resolution for the mm2pipeline_data package.

The data files (tilesets, converters, ...) live at the repo root, not in the
package, so everything resolves from there via repo_path().
"""
from pathlib import Path

import util.common_settings as common_settings

# mm2pipeline_data/paths.py -> mm2pipeline_data/ -> repo root
PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent


def repo_path(*parts) -> Path:
    """Path to a file/folder at the repository root (e.g. repo_path('smb.json'))."""
    return REPO_ROOT.joinpath(*parts)


# Canonical training tileset — the source of truth for the ASCII vocabulary.
# The location is defined once in util.common_settings 
MM2_TILESET_PATH = repo_path(*common_settings.MM2_TILESET.split("/"))

# Bundled toost.exe + its sprite/font assets.
TOOST_DIR = repo_path("MM2_Files", "toost_stuff")
