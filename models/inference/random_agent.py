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