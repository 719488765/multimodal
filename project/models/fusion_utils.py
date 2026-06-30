from models._load_pyc import load_model_pyc
_mod = load_model_pyc(__name__, "fusion_utils")
for _n, _v in _mod.__dict__.items():
    if not _n.startswith("_"):
        globals()[_n] = _v
