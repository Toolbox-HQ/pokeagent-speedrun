import os
import orjson

class JsonWriter:

    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path
        self.json = []

    def log(self, frame_num, key):
        self.json.append({"frame": frame_num, "keys": key})

    def close(self):
        with open(self.path, "wb") as f:
            f.write(orjson.dumps(self.json, option=orjson.OPT_INDENT_2))
