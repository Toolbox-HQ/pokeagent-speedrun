import multiprocessing as mp

def child_proc(connection, max_steps):
    from emulator.emulator_client import run
    run(connection=connection, max_steps=max_steps, manual_mode=False, agent_fps=2, save_state='./emulator/agent_direct_save.state')
    connection.close()

def main():
    from pathlib import Path
    import io
    from PIL import Image
    import numpy as np
    import torch
    from models.inference.agent_inference import Pokeagent
    
    agent = Pokeagent("cuda")


    # ctx = mp.get_context("spawn")
    # parent_conn, child_conn = ctx.Pipe(duplex=True)
    # MAX_STEPS = 100
    # c = ctx.Process(target=child_proc, args=(child_conn, MAX_STEPS))
    # c.start()

    # for i in range(MAX_STEPS):
    #     msg_type, payload = parent_conn.recv()
    #     assert msg_type == "image" and isinstance(payload, (bytes, bytearray))
    #     img = Image.open(io.BytesIO(payload)).convert("RGB")
    #     np_img = np.array(img)                           # HWC, uint8
    #     tensor = torch.from_numpy(np_img).permute(2, 0, 1).float() / 255.0
    #     action = agent.infer_action(tensor)
    #     parent_conn.send(("char", action))  # child now unblocks and continues
        
    # c.join()

if __name__ == "__main__":
    main()
