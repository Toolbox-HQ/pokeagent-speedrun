import multiprocessing as mp

def child_proc(connection, max_steps):
    from emulator.emulator_client import run
    run(connection=connection, max_steps=max_steps, manual_mode=False, agent_fps=2, save_state='./emulator/agent_direct_save.state')
    connection.close()

def main():
    ctx = mp.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=True)
    MAX_STEPS = 100
    c = ctx.Process(target=child_proc, args=(child_conn, MAX_STEPS))
    c.start()

    reply = ''
    for i in range(MAX_STEPS):
        msg_type, payload = parent_conn.recv()
        assert msg_type == "image" and isinstance(payload, (bytes, bytearray))
        
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

if __name__ == "__main__":
    main()
