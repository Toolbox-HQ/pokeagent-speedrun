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

    ctx = mp.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=True)
    MAX_STEPS = 100
    c = ctx.Process(target=child_proc, args=(child_conn, MAX_STEPS))
    c.start()

    last_img_bytes = None

    reply = ''
    for i in range(MAX_STEPS):
        msg_type, payload = parent_conn.recv()
        assert msg_type == "image" and isinstance(payload, (bytes, bytearray))
        last_img_bytes = payload
        
        if i % 6 == 0:
            reply = 'a'
        if i % 6 == 1:
            reply = 'b'
        if i % 6 == 2:
            reply = 'up'
        if i % 6 == 3:
            reply = 'down'
        if i % 6 == 4:
            reply = 'left'
        if i % 6 == 5:
            reply = 'right'
        
        parent_conn.send(("char", reply))  # child now unblocks and continues
    
    c.join()
    print("success!")

    Path("last_frame.png").write_bytes(last_img_bytes)
    img = Image.open(io.BytesIO(last_img_bytes)).convert("RGB")
    np_img = np.array(img)                           # HWC, uint8
    tensor = torch.from_numpy(np_img).permute(2, 0, 1).float() / 255.0
    print("tensor shape:", tuple(tensor.shape), "dtype:", tensor.dtype, "min/max:", float(tensor.min()), float(tensor.max()))


if __name__ == "__main__":
    main()
