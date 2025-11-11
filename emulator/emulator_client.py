import base64, io, threading, time
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, Response
import uvicorn
from emulator.mgba_emulator import MGBAEmulator
import argparse
import sys
import signal
import pygame
import cv2
import numpy as np
import json
import os
from typing import Union
from tqdm import tqdm 
from s3_utils.s3_sync import init_boto3_client, upload_to_s3

# Global state
latest_png_b64 = None
app = FastAPI()
emulator_fps = None
emulator = None
agent_mode = True
container = None
stream = None

# MP4 state
vw = None

# Key logging state
keys_json_path = None
keys_log_fp = None
_keys_log_first = True
frame_index = 0

curr_action = "none"

# Button mapping for manual control
button_map = {
    pygame.K_z: 'a',
    pygame.K_x: 'b', 
    pygame.K_RETURN: 'start',
    pygame.K_RSHIFT: 'select',
    pygame.K_UP: 'up',
    pygame.K_DOWN: 'down',
    pygame.K_LEFT: 'left',
    pygame.K_RIGHT: 'right',
}

# Pygame display
screen_width = 1920
screen_height = 1080
screen = None
font = None
clock = None

def init_pygame():
    """Initialize pygame"""
    global screen, font, clock
    
    pygame.init()
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Direct Agent Pokemon Emerald")
    font = pygame.font.Font(None, 24)
    clock = pygame.time.Clock()

def setup_emulator(rom_path, save_state:str =None):
    try:
        global emulator
        emulator = MGBAEmulator(rom_path)
        emulator.initialize()

        if save_state:
            emulator.load_state(save_state)

    except Exception as e:
            raise RuntimeError(f"Failed to initialize mgba: {e}")

def get_keys_for_frame(frame_interval, frame_index, connection):
    global curr_action
    actions_pressed = []

    if agent_mode:
        if frame_index % frame_interval == 1:
            msg_type, payload = connection.recv()
            if msg_type != "char":
                raise TypeError("Did not receive a char")
            curr_action = payload
        if curr_action != "none":
            actions_pressed.append(curr_action)
    else:
        # Make sure input state is fresh for this frame
        pygame.event.pump()
        keys = pygame.key.get_pressed()
        for key, action in button_map.items():
            if keys[key]:
                actions_pressed.append(action)
        if keys[pygame.K_1]:
            save_file = "agent_direct_save.state"
            emulator.save_state(save_file)
            print(f"State saved to: {save_file}")
        elif keys[pygame.K_2]:
            load_file = "agent_direct_save.state"
            if os.path.exists(load_file):
                emulator.load_state(load_file)
                print(f"State loaded from: {load_file}")

    return actions_pressed


def write_frame(frame):
    vw.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))

# NEW: streaming JSON logger
def _open_keys_logger(path):
    """Open the JSON file and start an array for streaming writes."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fp = open(path, "w", buffering=1)  # line-buffered
    fp.write("[\n")
    fp.flush()
    return fp

def map_key_to_str(action: list)-> str:
    if not action:
        return "none"
    else:
        return action[0]

def log_keys(frame_num, keys_list):
    """Append one frame's keys as a JSON object into the array."""
    global keys_log_fp, _keys_log_first
    if keys_log_fp is None:
        return
    entry = {"frame": frame_num, "keys": map_key_to_str(keys_list)}
    if not _keys_log_first:
        keys_log_fp.write(",\n")
    keys_log_fp.write(json.dumps(entry, ensure_ascii=False))
    keys_log_fp.flush()
    _keys_log_first = False

def _close_keys_logger():
    """Close the JSON array/file cleanly."""
    global keys_log_fp
    if keys_log_fp:
        keys_log_fp.write("\n]\n")
        keys_log_fp.flush()
        keys_log_fp.close()
        keys_log_fp = None

def game_loop(max_steps: int=0, connection = None, agent_fps = None) -> None:
    global latest_png_b64
    global frame_index
    print(f"[EMULATOR] begin emulation loop")
    if agent_mode:
        print(f"[EMULATOR] max_steps={max_steps}")

    frame_interval = int(round(60 / agent_fps))
    if agent_mode:
        pbar: tqdm = tqdm(total=max_steps) if max_steps else None

    while True:
        start = time.perf_counter()

        keys = get_keys_for_frame(frame_interval, frame_index, connection)

        log_keys(frame_index, keys)
       
        emulator.run_frame_with_keys(keys)
        frame = emulator.get_frame()


          # increment AFTER using this frame number
    
        if frame is None:
            print("frame is none")
            time.sleep(0.01)
            continue
        
        buf = io.BytesIO()
        frame.save(buf, format="PNG")
        latest_png_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        write_frame(frame)

        if agent_mode:
            if frame_index % frame_interval == 0:
                connection.send(("image", buf.getvalue()))
        else:
            surf = pygame.image.frombuffer(frame.tobytes(), frame.size, "RGB").convert()
            surf = pygame.transform.scale(surf, (screen_width, screen_height))
            screen.blit(surf, (0, 0))
            pygame.display.update()
        
        dt = time.perf_counter() - start
        
        if emulator_fps:
            frame_time = 1.0 / emulator_fps
            if dt < frame_time:
                time.sleep(frame_time - dt)

        frame_index += 1
        if agent_mode:
            if frame_index >= max_steps * frame_interval:
                break
            elif pbar and frame_index % frame_interval == 0:
                pbar.update(1)
                pbar.set_postfix({"MGBA ACTION" : keys})

@app.get("/")
def index():
    # Minimal page that polls /api/frame and shows it full-bleed
    html = """
    <!doctype html>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Live Frame</title>
    <style>
      html, body { margin: 0; padding: 0; }
      #screen {
        width: 60vw;     /* force full viewport width */
        height: auto;     /* keep aspect ratio */
        display: block;   /* remove inline gap */
        image-rendering: pixelated;
        image-rendering: crisp-edges; /* fallback */
      }
    </style>
    <img id="screen" alt="live frame" />
    <script>
      const img = document.getElementById('screen');
      function refresh() {
        img.src = '/api/frame?ts=' + Date.now(); // cache-bust per request
      }
      setInterval(refresh, 100); // ~10 fps (100 ms)
      refresh();
    </script>
    """
    return HTMLResponse(html)

@app.get("/api/frame")
def api_frame():
    b64 = latest_png_b64
    if not b64:
        return Response(status_code=503, headers={"Cache-Control": "no-store"})
    return Response(
        content=base64.b64decode(b64),
        media_type="image/png",
        headers={"Cache-Control": "no-store"},
    )

def run_fastapi_server(port):
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="error")

def save_to_s3(s3_path):
    s3 = init_boto3_client()
    bucket = s3.list_buckets()["Buckets"][0]["Name"]
    from pathlib import Path

    data_dir = Path("./emulator/data")

    for file in data_dir.iterdir():
        path = file.resolve()     
        file_name = file.name         
        upload_to_s3(path, f"pokeagent/{s3_path}/{file_name}", bucket, s3)

def quit(signum, _frame, save_s3: str = None):
    # Close video writer if initialized
    try:
        if vw is not None:
            vw.release()
    except Exception:
        pass
    # Close key logger cleanly
    try:
        _close_keys_logger()
    except Exception:
        pass

    if save_s3:
        save_to_s3(save_s3)

    sys.exit(0)

def run(rom: str = "./emulator/Emerald-GBAdvance/rom.gba", mp4_path: str = "./emulator/data/output.mp4", port: int = 8000, manual_mode: bool = False, fps: int = None, keys_json_path_local: str = "./emulator/data/keys.json", save_s3: str = None, save_state: str = None, max_steps: int = 3000, agent_fps: int = 2, connection = None):
    
    if manual_mode:
        global agent_mode
        agent_mode = False
        print("🎮 Starting in MANUAL mode (--manual-mode flag)")
        init_pygame()
    else:
        print("🤖 Starting in AGENT mode (default)")

    if save_state:
        assert(os.path.exists(save_state)), "Missing save state to load."

    # Set up signal handlers
    signal.signal(signal.SIGINT, quit)
    signal.signal(signal.SIGTERM, quit)

    setup_emulator(rom, save_state)
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # or 'avc1' if your build supports H.264
    global vw
    os.makedirs(os.path.dirname(mp4_path), exist_ok=True)
    vw = cv2.VideoWriter(mp4_path, fourcc, 60, (emulator.width, emulator.height))
    
    assert vw.isOpened(), "VideoWriter did not initialize correctly."
    
    # NEW: open the keys JSON logger
    global keys_json_path, keys_log_fp, _keys_log_first, frame_index
    keys_json_path = keys_json_path_local
    keys_log_fp = _open_keys_logger(keys_json_path)
    _keys_log_first = True
    frame_index = 0

    # Start web server in background thread
    server_thread = threading.Thread(target=run_fastapi_server, args=(port,), daemon=True)
    server_thread.start()
    
    global emulator_fps
    if fps:
        emulator_fps = fps
    try:
        game_loop(max_steps=max_steps, connection=connection, agent_fps=agent_fps) # Run main game loop
    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error: {e}")
    finally:
        quit(None, None, save_s3)


def main():
    parser = argparse.ArgumentParser(description="Direct Agent Pokemon Emerald")
    parser.add_argument("--rom", type=str, default="./emulator/Emerald-GBAdvance/rom.gba", help="Path to ROM file")
    parser.add_argument("--mp4-path", type=str, default="./emulator/data/output.mp4", help="Path to mp4 output")
    parser.add_argument("--port", type=int, default=8000, help="Port for web interface")
    parser.add_argument("--manual-mode", action="store_true", help="Start in manual mode instead of agent mode")
    parser.add_argument("--fps", type=int, help="Emulator fps (uncapped if not set)")
    parser.add_argument("--keys-json-path", type=str, default="./emulator/data/keys.json", help="Path to JSON file that logs per-frame keys")
    parser.add_argument("--save-s3", type=str, default=None, help="Save to s3 bucket, uses s3cmd credentials.")
    parser.add_argument("--save-state",type=str, default=None, help="Save state to start from.")
    parser.add_argument("--max-steps", type=int, default=3000, help="Maximum number of emulator steps.")
    parser.add_argument("--agent-fps", type=int, default=2, help="FPS the agent operates at when in agent mode.")
    args = parser.parse_args()
    run(args.rom, args.mp4_path, args.port, args.manual_mode, args.fps, args.keys_json_path, args.save_s3, args.save_state, args.max_steps, args.agent_fps)

if __name__ == "__main__":
   sys.exit(main())
