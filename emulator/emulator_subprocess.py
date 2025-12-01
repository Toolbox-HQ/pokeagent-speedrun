from emulator.mgba_emulator import MGBAEmulator
from emulator.keys import KEY_TO_MGBA
from emulator.json_writer import JsonWriter
import io
import cv2
import os
import numpy as np

EMULATOR = None
CONNECTION = None
KEY = "none"
FRAME = None
# VW = None # Video writer
# JW = None # Json writer
# FRAME_INDEX = -1
VwJwFrameIdxWritePairs = {}

def _initialize_emulator(rom_path, connection):
    global EMULATOR
    global CONNECTION
    EMULATOR = MGBAEmulator(rom_path)
    EMULATOR.initialize()
    CONNECTION = connection
    # os.makedirs(os.path.dirname(data_path + ".mp4"), exist_ok=True)
    # VW = cv2.VideoWriter(data_path + ".mp4", cv2.VideoWriter_fourcc(*"mp4v"), 60, (EMULATOR.width, EMULATOR.height))
    # assert VW.isOpened(), "VideoWriter did not initialize correctly."
    # JW = JsonWriter(data_path + ".json")
    _loop()

def _load_state(state_bytes: bytes):
    EMULATOR.load_state_bytes(state_bytes)

def _get_state():
    CONNECTION.send(("state", EMULATOR.get_state_bytes()))

def _set_key(key: str):
    global KEY
    KEY = key

def _run_frames(num_frames: int):
    global FRAME
    for i in range(num_frames):
        EMULATOR.run_frame_with_keys(KEY_TO_MGBA[KEY])
        FRAME = EMULATOR.get_frame()
        for key, value in VwJwFrameIdxWritePairs.items():
            if value['write']:
                value['idx'] += 1
                value['vw'].write(cv2.cvtColor(np.array(FRAME), cv2.COLOR_RGB2BGR))
                value['jw'].log(value['idx'], KEY)

def _get_current_frame():
    if FRAME is None: 
        _run_frames(1)
    buffer = io.BytesIO()
    FRAME.save(buffer, format="PNG")
    CONNECTION.send(("image", buffer.getvalue()))

def _quit():
    for key, value in VwJwFrameIdxWritePairs:
        value['vw'].release()
        value['jw'].close()
    EMULATOR.stop()

def _create_video_writer(path: str):
    os.makedirs(os.path.dirname(path + ".mp4"), exist_ok=True)
    vw = cv2.VideoWriter(path + ".mp4", cv2.VideoWriter_fourcc(*"mp4v"), 60, (EMULATOR.width, EMULATOR.height))
    assert vw.isOpened(), "VideoWriter did not initialize correctly."
    jw = JsonWriter(path + ".json")
    frame_idx = -1
    write = False
    if path in VwJwFrameIdxWritePairs:
        raise Exception(f"video writer path conflict: {path}")
    VwJwFrameIdxWritePairs[path] = {'vw': vw, 'jw': jw, 'idx': frame_idx, 'write': write}

def _start_video_writer(path: str):
    VwJwFrameIdxWritePairs[path]['write'] = True

def _pause_video_writer(path: str):
    VwJwFrameIdxWritePairs[path]['write'] = False

def _release_video_writer(path: str):
    VwJwFrameIdxWritePairs[path]['vw'].release()
    VwJwFrameIdxWritePairs[path]['jw'].close()
    del VwJwFrameIdxWritePairs[path]

def _loop():
    while True:
        msg_type, payload = CONNECTION.recv()
        match msg_type:
            case "load_state":
                _load_state(payload)
            case "get_state":
                _get_state()
            case "set_key":
                _set_key(payload)
            case "run_frames":
                _run_frames(payload)
            case "get_current_frame":
                _get_current_frame()
            case "create_video_writer":
                _create_video_writer(payload)
            case "start_video_writer":
                _start_video_writer(payload)
            case "pause_video_writer":
                _pause_video_writer(payload)
            case "release_video_writer":
                _release_video_writer(payload)
            case "quit":
                _quit()
                break