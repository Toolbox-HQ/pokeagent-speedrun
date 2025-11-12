import os
import json

class JsonWriter:

    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fp = open(path, "w", buffering=1)  # line-buffered
        fp.write("[\n")
        fp.flush()
        self.fp = fp

    def log(self, frame_num, key):
        entry = {"frame": frame_num, "keys": key}
        self.fp.write(json.dumps(entry, ensure_ascii=False))
        self.fp.flush()

    def close(self):
        self.fp.write("\n]\n")
        self.fp.flush()
        self.fp.close()