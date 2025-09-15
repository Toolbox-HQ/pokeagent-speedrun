import base64, io, threading, time
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, Response
import uvicorn
from emerald_emulator_mgba import EmeraldEmulator  # keep OR replace with your local class
import argparse
import sys
import signal
import pygame
import cv2
from PIL import Image
import numpy as np
import json
import os

# Global state
latest_png_b64 = None
app = FastAPI()
emulator_fps = None
emulator = None
policy = None
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
screen_width = 480  # 240 * 2 (upscaled)
screen_height = 320  # 160 * 2 (upscaled)
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

def setup_emulator(rom_path="Emerald-GBAdvance/rom.gba", load_state=None):
    try:
        global emulator
        emulator = EmeraldEmulator(rom_path)
        emulator.initialize()
    except Exception as e:
            raise RuntimeError(f"Failed to initialize mgba: {e}")

def get_keys_for_frame():
    actions_pressed = []
    if agent_mode:
        return actions_pressed

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

def log_keys(frame_num, keys_list):
    """Append one frame's keys as a JSON object into the array."""
    global keys_log_fp, _keys_log_first
    if keys_log_fp is None:
        return
    entry = {"frame": frame_num, "keys": keys_list}
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

def game_loop(max_steps: int=0) -> None:
    global latest_png_b64, frame_index
    
    while True:
        try:
            
            start = time.perf_counter()

            # 1) gather input keys for this frame
            keys = get_keys_for_frame()

            # 2) log them with the current frame index
            log_keys(frame_index, keys)

            # 3) advance emulator with those keys
            emulator.run_frame_with_keys(keys)
            frame = emulator.get_frame()
            frame_index += 1  # increment AFTER using this frame number

            if frame is None:
                time.sleep(0.01)
                continue
            
            buf = io.BytesIO()
            frame.save(buf, format="PNG")
            latest_png_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            write_frame(frame)
            dt = time.perf_counter() - start
            
            if emulator_fps:
                frame_time = 1.0 / emulator_fps
                if dt < frame_time:
                    time.sleep(frame_time - dt)

            if max_steps and frame_index >= max_steps:
                break

        except Exception as e:
            # log and keep going
            print(f"emulator_loop error: {e}")
            time.sleep(0.5)

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

def quit(signum, _frame):
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
    sys.exit(0)

def main():
    parser = argparse.ArgumentParser(description="Direct Agent Pokemon Emerald")
    parser.add_argument("--rom", type=str, default="Emerald-GBAdvance/rom.gba", help="Path to ROM file")
    parser.add_argument("--mp4-path", type=str, default="./Data/output.mp4", help="Path to mp4 output")
    parser.add_argument("--port", type=int, default=8000, help="Port for web interface")
    parser.add_argument("--manual-mode", action="store_true", help="Start in manual mode instead of agent mode")
    parser.add_argument("--fps", type=int, help="Emulator fps (uncapped if not set)")
    parser.add_argument("--keys-json-path", type=str, default="Data/keys.json", help="Path to JSON file that logs per-frame keys")  # NEW
    parser.add_argument("--max-steps", type=int, default=0, help="Number of emulator steps before the emulator quits.")

    args = parser.parse_args()

    if args.manual_mode:
        global agent_mode
        agent_mode = False
        print("🎮 Starting in MANUAL mode (--manual-mode flag)")
    else:
        print("🤖 Starting in AGENT mode (default)")
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, quit)
    signal.signal(signal.SIGTERM, quit)

    if args.manual_mode:
        init_pygame()

    setup_emulator(args.rom, None)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # or 'avc1' if your build supports H.264
    global vw
    os.makedirs(os.path.dirname(args.mp4_path), exist_ok=True)
    vw = cv2.VideoWriter(args.mp4_path, fourcc, 60, (emulator.width, emulator.height))
    
    assert vw.isOpened(), "VideoWriter did not initialize correctly."

    # NEW: open the keys JSON logger
    global keys_json_path, keys_log_fp, _keys_log_first, frame_index
    keys_json_path = args.keys_json_path
    keys_log_fp = _open_keys_logger(keys_json_path)
    _keys_log_first = True
    frame_index = 0

    # Start web server in background thread
    server_thread = threading.Thread(target=run_fastapi_server, args=(args.port,), daemon=True)
    server_thread.start()

    global emulator_fps
    if args.fps:
        emulator_fps = args.fps
    try:
        game_loop(max_steps=args.max_steps) # Run main game loop
    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        quit(None, None)

if __name__ == "__main__":
   sys.exit(main())
