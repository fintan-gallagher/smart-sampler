import json, os

PREFS_PATH = os.path.expanduser('~/.smart_sampler_prefs.json')

def load_prefs() -> dict:
    try:
        with open(PREFS_PATH) as f:
            return json.load(f)
    except Exception:
        return {}

def save_prefs(data: dict):
    try:
        with open(PREFS_PATH, 'w') as f:
            json.dump(data, f)
    except Exception:
        pass