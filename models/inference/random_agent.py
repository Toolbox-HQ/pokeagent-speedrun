import random

class RandomAgent:

    def __init__(self, min_frames, max_frames, key_map):
        self.min_frames = min_frames
        self.max_frames = max_frames
        self.key_map = key_map
    
    def infer(self):
        num_frames = random.randint(self.min_frames, self.max_frames)
        key = random.choice(self.key_map)
        return key, num_frames

class LZRandomAgent:

    def __init__(self, min_frames, max_frames, key_map):
        self.min_frames = min_frames
        self.max_frames = max_frames
        self.key_map = key_map
        self.prev_frame = None
    
    ROLL_REQUIRES = {
        "up+r": "up",
        "down+r": "down",
        "left+r": "left",
        "right+r": "right",
    }

    def infer(self):
        num_frames = random.randint(self.min_frames, self.max_frames)
        available = [k for k in self.key_map
                     if k not in self.ROLL_REQUIRES
                     or self.ROLL_REQUIRES[k] == self.prev_frame]
        key = random.choice(available)
        self.prev_frame = key
        return key, num_frames