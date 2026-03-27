import argparse
import base64
from io import BytesIO
from emulator.keys import KEY_LIST_FOR_TRAINING
import json
import os

SYSTEM_PROMPT = """\
You are playing Pokémon Emerald on a Game Boy Advance emulator.
You will be shown the current game screen and must choose the single best action to make progress in the game.
You MUST call both tools every step: press_button AND update_memory.

Button reference:
- up / down / left / right: Move the character on the overworld; navigate cursor in menus and battle.
- a: Confirm a menu selection; interact with NPCs, signs, and objects; advance dialogue; select a move in battle.
- b: Cancel or close a menu; hold to run (after obtaining Running Shoes); skip or speed through dialogue.
- start: Open the main pause menu (Pokémon, Bag, Save, etc.).
- select: Register an item for quick use (after unlocking); swap menu shortcuts.
- none: Do nothing this frame (use when waiting for an animation or cutscene to finish).

Memory instructions:
- Your persistent memory is shown below the screenshot each step.
- After every step you MUST call update_memory with whatever notes you want carried forward.
- Use memory to track your location, current objective, party state, and any important observations.\
"""

USER_TEMPLATE = """\
Current memory:
{memory}

What is the best action to take next?\
"""

PRESS_BUTTON_TOOL = {
    "type": "function",
    "function": {
        "name": "press_button",
        "description": "Press a button on the Game Boy Advance to control the game.",
        "parameters": {
            "type": "object",
            "properties": {
                "button": {
                    "type": "string",
                    "enum": KEY_LIST_FOR_TRAINING,
                    "description": "The button to press.",
                }
            },
            "required": ["button"],
        },
    },
}

UPDATE_MEMORY_TOOL = {
    "type": "function",
    "function": {
        "name": "update_memory",
        "description": (
            "Persist notes to carry forward into the next step. "
            "Overwrite the full memory each call — anything not included is forgotten. "
            "Track location, current objective, party state, and key observations."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "memory": {
                    "type": "string",
                    "description": "Free-form text to remember for the next step.",
                }
            },
            "required": ["memory"],
        },
    },
}

TOOLS = [PRESS_BUTTON_TOOL, UPDATE_MEMORY_TOOL]


def frame_to_b64(frame) -> str:
    buf = BytesIO()
    frame.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def build_messages(frame, memory: str) -> list:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{frame_to_b64(frame)}"},
                },
                {"type": "text", "text": USER_TEMPLATE.format(memory=memory or "(empty)")},
            ],
        },
    ]


def parse_output(output) -> tuple[str, str]:
    """Returns (action, memory) parsed from tool calls."""
    action = "none"
    memory = None
    tool_calls = getattr(output, "tool_calls", None)
    if tool_calls:
        for call in tool_calls:
            try:
                args = json.loads(call.function.arguments)
            except (json.JSONDecodeError, AttributeError):
                continue
            if call.function.name == "press_button":
                button = args.get("button", "none")
                if button in KEY_LIST_FOR_TRAINING:
                    action = button
            elif call.function.name == "update_memory":
                memory = args.get("memory", "")
    if action == "none":
        # Fallback: scan raw text
        text = (output.text or "").strip().lower()
        for key in KEY_LIST_FOR_TRAINING:
            if key in text:
                action = key
                break
    return action, memory or ""


def main():

    os.environ.setdefault("VLLM_CACHE_ROOT", "./tmp/vllm")
    import torch
    from vllm import LLM, SamplingParams
    from emulator.emulator_connection import EmulatorConnection
    from models.util.misc import local_model_map

    parser = argparse.ArgumentParser(description="vLLM QwenVL baseline for Pokémon Emerald")
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--rom", default=".cache/pokeagent/rom/rom.gba")
    parser.add_argument("--save-state", default=".cache/pokeagent/save_state/truck_start.state")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--video-out", default="./tmp/out", help="Path prefix for output video (omit extension)")
    parser.add_argument("--log-out", default="./log.json", help="Path for JSON step log")
    args = parser.parse_args()

    for path in [args.rom, args.save_state]:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Required file not found: {path}")

    print(f"Connecting to emulator: {args.rom}")
    conn = EmulatorConnection(args.rom)
    conn.load_state_from_file(args.save_state)

    print(f"Loading model: {args.model}")
    llm = LLM(
        model=local_model_map(args.model),
        limit_mm_per_prompt={"image": 1},
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=2048,
    )
    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=0.95,
        top_k=20,
        min_p=0.0,
        presence_penalty=1.5,
        repetition_penalty=1.0,
        max_tokens=args.max_tokens,
    )
    if args.video_out:
        os.makedirs(os.path.dirname(args.video_out) or ".", exist_ok=True)
        conn.create_video_writer(args.video_out)
        conn.start_video_writer(args.video_out)

    memory: str = ""
    log: list = []

    os.makedirs(os.path.dirname(args.log_out) or ".", exist_ok=True)

    print(f"Running for {args.steps} steps")
    for step in range(args.steps):
        print("[GET FRAME FROM EMULATOR]")
        frame = conn.get_current_frame()
        frame_b64 = frame_to_b64(frame)
        messages = build_messages(frame, memory)
        print("[DO CHAT]")
        outputs = llm.chat(
            messages,
            sampling_params=sampling_params,
            tools=TOOLS,
            chat_template_kwargs={"tool_choice": "required"},
        )
        output = outputs[0].outputs[0]
        print(f"[RAW CHAT] text={repr(output.text)} tool_calls={getattr(output, 'tool_calls', None)}")
        action, memory = parse_output(output)

        tool_calls_log = None
        raw_tool_calls = getattr(output, "tool_calls", None)
        if raw_tool_calls:
            tool_calls_log = [
                {"name": tc.function.name, "arguments": tc.function.arguments}
                for tc in raw_tool_calls
            ]

        log.append({
            "step": step,
            "messages": [
                {k: v for k, v in m.items() if k != "content"}
                | {"content": [
                    ({"type": c["type"], "image_b64": "<omitted>"} if c.get("type") == "image_url" else c)
                    for c in m["content"]
                ] if isinstance(m["content"], list) else m["content"]}
                for m in messages
            ],
            "frame_b64": frame_b64,
            "raw_text": output.text,
            "tool_calls": tool_calls_log,
            "action": action,
            "memory": memory,
            "finish_reason": output.finish_reason,
        })
        with open(args.log_out, "w") as f:
            json.dump(log, f, indent=2)

        print("[SET KEY]")
        conn.set_key(action)
        print("[RUN FRAMES]")
        conn.run_frames(15)

        print(f"Step {step}/{args.steps} | action={action} | memory={repr(memory)}")

    if args.video_out:
        conn.release_video_writer(args.video_out)
    conn.close()
    print("Done.")


if __name__ == "__main__":
    main()
