import os

# This is the format for MGBA emulator
MGBA_KEY_LIST = [
    ["a"],
    ["b"],
    ["start"],
    ["select"],
    ["up"],
    ["down"],
    ["left"],
    ["right"],
    [],
]

# This is the format we use for logging / training
KEY_LIST_FOR_TRAINING = [
    "a",
    "b",
    "start",
    "select",
    "up",
    "down",
    "left",
    "right",
    "none",
]

KEY_LIST_FOR_IDM = [
    "a",
    "b",
    "start",
    "up",
    "down",
    "left",
    "right",
    "none",
]



# ── lz mode (Legend of Zelda) ─────────────────────────────────────────────────

MGBA_KEY_LIST_LZ = [
    ["a"],
    ["b"],
    ["start"],
    ["select"],
    ["up"],
    ["down"],
    ["left"],
    ["right"],
    ["up", "right"],
    ["up", "left"],
    ["down", "right"],
    ["down", "left"],
    ["up", "r"],
    ["right", "r"],
    ["left", "r"],
    ["down", "r"],
    [],
]

KEY_LIST_FOR_TRAINING_LZ = [
    "a",
    "b",
    "start",
    "select",
    "up",
    "down",
    "left",
    "right",
    "up+right",
    "up+left",
    "down+right",
    "down+left",
    "up+r",
    "right+r",
    "left+r",
    "down+r",
    "none",
]

KEY_LIST_FOR_IDM_LZ = [
    "a",
    "b",
    "start",
    "up",
    "down",
    "left",
    "right",
    "up+right",
    "up+left",
    "down+right",
    "down+left",
    "up+r",
    "right+r",
    "left+r",
    "down+r",
    "none",
]

# set things for lz
if "LZ_MODE" in os.environ:
    MGBA_KEY_LIST = MGBA_KEY_LIST_LZ
    KEY_LIST_FOR_TRAINING = KEY_LIST_FOR_TRAINING_LZ
    KEY_LIST_FOR_IDM = KEY_LIST_FOR_IDM_LZ

CLASS_TO_KEY = {ind:key for (ind, key) in enumerate(KEY_LIST_FOR_TRAINING)}
KEY_TO_CLASS = {v:k for k,v in CLASS_TO_KEY.items()}
KEY_TO_MGBA = {v:k for (v,k) in zip(KEY_LIST_FOR_TRAINING, MGBA_KEY_LIST)}
NUM_ACTION_CLASSES = len(CLASS_TO_KEY)
