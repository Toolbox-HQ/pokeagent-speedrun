from emulator.mgba_emulator import MGBAEmulator

EMULATOR = None
CONNECTION = None
ACTION = None
FRAME = None

def initialize_emulator(rom_path, connection):
    global EMULATOR
    global CONNECTION
    EMULATOR = MGBAEmulator(rom_path)
    EMULATOR.initialize()
    CONNECTION = connection
    _loop()

def _load_state(state_bytes):
    EMULATOR.load_state_bytes(state_bytes)

def _get_state():
    CONNECTION.send("state", EMULATOR.get_state_bytes())

def _set_action(action):
    global ACTION
    ACTION = action

def _run_frames(num_frames):
    for i in range()


def _loop():
    while True:
        msg_type, payload = CONNECTION.recv()
        match msg_type:
            case "load_state":
                _load_state(payload)
            case "get_state":
                _get_state()
            case "set_action":
                _set_action(payload)




