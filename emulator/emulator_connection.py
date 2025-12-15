import multiprocessing as mp
from PIL import Image
import io

class EmulatorConnection:

    def __init__(self, rom_path):
        ctx = mp.get_context("spawn")
        self.parent_conn, child_conn = ctx.Pipe(duplex=True)
        self._child = ctx.Process(target=self._start_subprocess, args=(rom_path, child_conn))
        self._child.start()
        child_conn.close()

    @staticmethod
    def _start_subprocess(rom_path, conn):
        from emulator.emulator_subprocess import _initialize_emulator
        _initialize_emulator(rom_path, conn)
    
    def load_state(self, state_bytes: bytes):
        self.parent_conn.send(("load_state", state_bytes))
    
    def get_state(self) -> bytes:
        self.parent_conn.send(("get_state", None))
        msg_type, payload = self.parent_conn.recv()
        assert msg_type == "state"
        return payload
    
    def save_state(self, path: str) -> None:
        import os

        bytes = self.get_state()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(bytes)

    def load_state_from_file(self, path: str) -> None:
        with open(path, "rb") as f:
            state_bytes = f.read()
        self.load_state(state_bytes)

    def set_key(self, key: str):
        self.parent_conn.send(("set_key", key))
    
    def run_frames(self, num_frames: int):
        self.parent_conn.send(("run_frames", num_frames))
    
    def get_current_frame(self) -> Image:
        self.parent_conn.send(("get_current_frame", None))
        msg_type, payload = self.parent_conn.recv()
        assert msg_type == "image"
        img = Image.open(io.BytesIO(payload))
        img = img.convert("RGB")
        return img

    def close(self):
        self.parent_conn.send(("quit", None))
        self._child.join()

    def create_video_writer(self, path: str):
        self.parent_conn.send(("create_video_writer", path))
    
    def start_video_writer(self, path: str):
        self.parent_conn.send(("start_video_writer", path))

    def pause_video_writer(self, path: str):
        self.parent_conn.send(("pause_video_writer", path))

    def release_video_writer(self, path: str):
        self.parent_conn.send(("release_video_writer", path))