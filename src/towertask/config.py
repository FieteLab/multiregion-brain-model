from pathlib import Path

# Detect repo root
try:
    REPO_ROOT = Path(__file__).resolve().parents[2]
except NameError:
    REPO_ROOT = Path.cwd()

###################################################################
# Save all plots under icml-camera-ready/ - user may override this if needed
FIGURE_DIR = str(REPO_ROOT / "icml-results")
Path(FIGURE_DIR).mkdir(parents=True, exist_ok=True)

# Default data/cache directory — users may override this if needed
# For example:
# DATA_DIR = "/om2/user/xieyi/icml-data"

DATA_DIR = str(REPO_ROOT / "icml-data")
Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
###################################################################

def get_figure_path(fig_name: str, fname: str) -> str:
    """
    Returns the full path to save a figure file, creating subdirectories as needed.

    Parameters
    ----------
    fig_name : str
        Name of the figure or subdirectory to organize related plots (e.g., "Fig3a").
    fname : str
        Filename for the plot (e.g., "LE_alpha1.5_g0.80.png").

    Returns
    -------
    str
        Full path under FIGURE_DIR where the figure should be saved.
    """
    path = Path(FIGURE_DIR) / fig_name
    path.mkdir(parents=True, exist_ok=True)
    return str(path / fname)
