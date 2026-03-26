import argparse
import base64
import os
from collections import deque
from io import BytesIO
from models.util.misc import local_model_map
from vllm import LLM, SamplingParams

from emulator.emulator_connection import EmulatorConnection
from emulator.keys import KEY_LIST_FOR_TRAINING

SYSTEM_PROMPT = """\
You are playing Pokémon Emerald on a Game Boy Advance emulator.
You will be shown the current game screen and a history of recent actions.
Your task is to choose the single best action to make progress in the game.

Valid actions: a, b, start, select, up, down, left, right, none

Rules:
- Output exactly one action word and nothing else.
- Use directional keys (up, down, left, right) to navigate menus and move the character.
- Use 'a' to confirm selections and interact with NPCs and objects.
- Use 'b' to cancel or go back.
- Use 'start' to open the menu.
- Use 'none' if no action is needed.\
"""

USER_TEMPLATE = """\
Recent actions (oldest to newest): {action_history}

What is the best action to take next?\
"""


def frame_to_b64(frame) -> str:
    buf = BytesIO()
    frame.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def build_messages(frame, action_history: deque) -> list:
    history_str = ", ".join(action_history) if action_history else "none"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{frame_to_b64(frame)}"},
                },
                {"type": "text", "text": USER_TEMPLATE.format(action_history=history_str)},
            ],
        },
    ]


def parse_action(text: str) -> str:
    text = text.strip().lower()
    for action in KEY_LIST_FOR_TRAINING:
        if action in text:
            return action
    return "none"


def main():
    parser = argparse.ArgumentParser(description="vLLM QwenVL baseline for Pokémon Emerald")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--rom", default=".cache/pokeagent/rom/rom.gba")
    parser.add_argument("--save-state", default=".cache/pokeagent/save_state/truck_start.state")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--context-length", type=int, default=16, help="Number of past actions to include in prompt")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--video-out", default="./tmp/out", help="Path prefix for output video (omit extension)")
    args = parser.parse_args()

    for path in [args.rom, args.save_state]:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Required file not found: {path}")

    os.environ.setdefault("VLLM_CACHE_ROOT", ".cache/pokeagent/vllm")

    print(f"Connecting to emulator: {args.rom}")
    conn = EmulatorConnection(args.rom)
    

    print(f"Loading model: {args.model}")
    llm = LLM(
        model=args.model,
        limit_mm_per_prompt={"image": 1},
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    sampling_params = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens)

    print("main: loading state")
    conn.load_state_from_file(args.save_state)
    print("main: finished loading state")
    if args.video_out: 
        os.makedirs(os.path.dirname(args.video_out) or ".", exist_ok=True)
        conn.create_video_writer(args.video_out)
        conn.start_video_writer(args.video_out)

    action_history: deque = deque(maxlen=args.context_length)

    print(f"Running for {args.steps} steps")
    for step in range(args.steps):
        frame = conn.get_current_frame()
        messages = build_messages(frame, action_history)
        outputs = llm.chat(messages, sampling_params=sampling_params)
        raw = outputs[0].outputs[0].text
        action = parse_action(raw)
        action_history.append(action)

        conn.run_frames(7)
        conn.set_key(action)
        conn.run_frames(8)

        if step % 100 == 0:
            print(f"Step {step}/{args.steps} | action={action} | raw={repr(raw)}")

    if args.video_out:
        conn.release_video_writer(args.video_out)
    conn.close()
    print("Done.")


if __name__ == "__main__":
    main()
