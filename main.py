def main():
    from pathlib import Path
    import io
    from PIL import Image
    import numpy as np
    import torch
    from models.inference.agent_inference import Pokeagent
    from emulator.emulator_connection import EmulatorConnection
    from tqdm import tqdm
    
    agent = Pokeagent(device="cuda", temperature=1)
    conn = EmulatorConnection("emulator/Emerald-GBAdvance/rom.gba", "emulator/data/output")
    with open("emulator/agent_direct_save.state", 'rb') as f:
        state_bytes = f.read()
    conn.load_state(state_bytes)
    MAX_STEPS = 100
    for i in tqdm(range(MAX_STEPS)):
        tensor = torch.from_numpy(np.array(conn.get_current_frame())).permute(2, 0, 1) # CHW, uint8
        key = agent.infer_action(tensor)
        conn.set_key(key)
        conn.run_frames(30)
    conn.close()

if __name__ == "__main__":
    main()
