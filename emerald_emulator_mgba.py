import logging
from pathlib import Path
import tempfile
import mgba.core
import mgba.log
import mgba.image
from mgba._pylib import ffi, lib
import queue
from typing import List

logger = logging.getLogger(__name__)

class EmeraldEmulator:
    """emulator wrapper for Pokémon Emerald with headless frame capture and scripted inputs."""

    def __init__(self, rom_path: str):
        self.rom_path = rom_path
        self.core = None
        self.width = 240
        self.height = 160

        # Define key mapping for mgba
        self.KEY_MAP = {
            "a": lib.GBA_KEY_A,
            "b": lib.GBA_KEY_B,
            "start": lib.GBA_KEY_START,
            "select": lib.GBA_KEY_SELECT,
            "up": lib.GBA_KEY_UP,
            "down": lib.GBA_KEY_DOWN,
            "left": lib.GBA_KEY_LEFT,
            "right": lib.GBA_KEY_RIGHT,
            "l": lib.GBA_KEY_L,
            "r": lib.GBA_KEY_R,
        }

    def initialize(self):
        """Load ROM and set up emulator"""
        try:
            # Prevents relentless spamming to stdout by libmgba.
            mgba.log.silence()
            
            # Create a temporary directory and copy the gba file into it
            # this is necessary to prevent mgba from overwriting the save file (and to prevent crashes)
            tmp_dir = Path(tempfile.mkdtemp())
            tmp_gba = tmp_dir / "rom.gba"
            tmp_gba.write_bytes(Path(self.rom_path).read_bytes())
            
            # Load the core
            self.core = mgba.core.load_path(str(tmp_gba))
            if self.core is None:
                raise ValueError(f"Failed to load GBA file: {self.rom_path}")
            
            # Get dimensions from the core
            self.width, self.height = self.core.desired_video_dimensions()
            logger.info(f"mgba initialized with ROM: {self.rom_path} and dimensions: {self.width}x{self.height}")
            
            # Set up video buffer for frame capture using mgba.image.Image
            self.video_buffer = mgba.image.Image(self.width, self.height)
            self.core.set_video_buffer(self.video_buffer)
            self.core.reset()  # Reset after setting video buffer
            
            logger.info(f"mgba initialized with ROM: {self.rom_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize mgba: {e}")
        
    def get_frame(self):
        """Return the current frame as a PIL image"""
        if not self.core or not self.video_buffer:
            return None
        
        try:
            # Use the built-in to_pil() method from mgba.image.Image
            if hasattr(self.video_buffer, 'to_pil'):
                screenshot = self.video_buffer.to_pil()
                if screenshot:
                    screenshot = screenshot.convert("RGB")
                    return screenshot
                else:
                    logger.warning("mgba.image.Image does not have to_pil method")
                    return None
            else:
                logger.warning("mgba.image.Image does not have to_pil method")
                return None
        except Exception as e:
            logger.error(f"Failed to get screenshot: {e}")
            return None
    
    def run_frame_with_keys(self, key_strings: List[str]):
        if len(key_strings) > 0:
            keys = [self.KEY_MAP[key] for key in key_strings]
            for key in keys:
                self.core.add_keys(key)
            self.core.run_frame()
            for key in keys:
                self.core.clear_keys(key)
        else:
            self.core.run_frame()
    
    def load_state(self, path: str):
        """Load emulator state from file or memory"""
        if not self.core:
            return
        try:
            if path:
                with open(path, 'rb') as f:
                    state_bytes = f.read()
                    if state_bytes:
                        # Ensure state_bytes is actually bytes
                        if not isinstance(state_bytes, bytes):
                            state_bytes = bytes(state_bytes)
                        self.core.load_raw_state(state_bytes)
                        # Run a frame to ensure memory is properly loaded
                        self.core.run_frame()
        except Exception as e:
            return
        
    def save_state(self, path: str):
        """Save current emulator state to file or return as bytes"""
        if not self.core:
            return None
        
        try:
            # Get the raw state data
            raw_data = self.core.save_raw_state()
            
            # Convert CFFI object to bytes if needed
            if hasattr(raw_data, 'buffer'):
                data = bytes(raw_data.buffer)
            elif hasattr(raw_data, '__len__'):
                data = bytes(raw_data)
            else:
                data = raw_data
            
            if path:
                with open(path, 'wb') as f:
                    f.write(data)
            
            return data
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
            return None