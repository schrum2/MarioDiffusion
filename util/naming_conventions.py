model_name_map = [
    ("Mar1and2-conditional-regular", "MLM-regular"),
    ("Mar1and2-conditional-absence", "MLM-absence"),
    ("Mar1and2-conditional-negative", "MLM-negative"),
    ("Mar1and2-conditional-MiniLM-regular", "MiniLM-single-regular"),
    ("Mar1and2-conditional-MiniLM-absence", "MiniLM-single-absence"),
    ("Mar1and2-conditional-MiniLM-negative", "MiniLM-single-negative"),
    ("Mar1and2-conditional-MiniLMsplit-regular", "MiniLM-multiple-regular"),
    ("Mar1and2-conditional-MiniLMsplit-absence", "MiniLM-multiple-absence"),
    ("Mar1and2-conditional-MiniLMsplit-negative", "MiniLM-multiple-negative"),
    ("Mar1and2-conditional-GTE-regular", "GTE-single-regular"),
    ("Mar1and2-conditional-GTE-absence", "GTE-single-absence"),
    ("Mar1and2-conditional-GTE-negative", "GTE-single-negative"),
    ("Mar1and2-conditional-GTEsplit-regular", "GTE-multiple-regular"),
    ("Mar1and2-conditional-GTEsplit-absence", "GTE-multiple-absence"),
    ("Mar1and2-conditional-GTEsplit-negative", "GTE-multiple-negative"),
    ("Mar1and2-fdm-MiniLM-regular", "FDM-MiniLM-regular"),
    ("Mar1and2-fdm-MiniLM-absence", "FDM-MiniLM-absence"),
    ("Mar1and2-fdm-GTE-regular", "FDM-GTE-regular"),
    ("Mar1and2-fdm-GTE-absence", "FDM-GTE-absence"),
    ("Mar1and2-wgan", "WGAN"),
    ("Mar1and2-unconditional", "Unconditional"),
    ("MarioGPT_metrics", "MarioGPT"),
]

def get_model_name_map_and_order():
    mapping = dict(model_name_map)
    order = [v for k, v in model_name_map]
    return mapping, order