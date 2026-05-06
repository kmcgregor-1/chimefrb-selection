from .utils import PROP_ORDER, data_dir, DEFAULT_MODELS_DIR, DEFAULT_KNN_DIR
from .funcs import build_design_matrix, logistic_selection, logistic_selection_batch
from .selection import SelectionFunction
from .knn_mask import KNNEnvelope

# Version (optional)
try:
    from ._version import __version__
except Exception:
    __version__ = "0.1.0"

__all__ = [
    "PROP_ORDER", "data_dir", "DEFAULT_MODELS_DIR", "DEFAULT_KNN_DIR",
    "build_design_matrix", "logistic_selection", "logistic_selection_batch",
    "SelectionFunction", "KNNEnvelope",
    "__version__",
]