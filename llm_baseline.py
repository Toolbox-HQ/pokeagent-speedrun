import argparse
import base64
import collections
from datetime import datetime
from io import BytesIO
from emulator.keys import KEY_LIST_FOR_TRAINING
import json
import os
import re

FRAME_HISTORY_SPACING = 30   # env steps between context frames (~0.5 s)
FRAME_HISTORY_COUNT = 64     # number of context frames to include

SYSTEM_PROMPT = """\
You are playing Pokémon Emerald on a Game Boy Advance emulator.
You will be shown a sequence of up to 64 screenshots captured every ~0.5 seconds of game time, \
ordered from oldest to most recent. The last image is the current frame. \
Use the sequence to understand what has been happening and choose the single best next action.
You must call the press_button tool every step.

Button reference:
- up / down / left / right: Move the character on the overworld; navigate cursor in menus and battle.
- a: Confirm a menu selection; interact with NPCs, signs, and objects; advance dialogue; select a move in battle.
- b: Cancel or close a menu; hold to run (after obtaining Running Shoes); skip or speed through dialogue.
- start: Open the main pause menu (Pokémon, Bag, Save, etc.).
- select: Register an item for quick use (after unlocking); swap menu shortcuts.
- none: Do nothing this frame (use when waiting for an animation or cutscene to finish).\
"""

USER_TEMPLATE = """\
The {n_frames} image(s) above show the last ~{window_secs:.0f} seconds of gameplay (oldest → most recent).
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

TOOLS = [PRESS_BUTTON_TOOL]


def frame_to_b64(frame) -> str:
    buf = BytesIO()
    frame.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def build_messages(frames) -> list:
    """Build chat messages with a temporal context window of frames.

    ``frames`` is a list of PIL images in chronological order (oldest first,
    most recent last).  All frames are included as image_url blocks followed by
    the text prompt.
    """
    image_blocks = [
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{frame_to_b64(f)}"},
        }
        for f in frames
    ]
    n_frames = len(frames)
    window_secs = (n_frames - 1) * FRAME_HISTORY_SPACING / 60  # 60 env steps per second
    text = USER_TEMPLATE.format(n_frames=n_frames, window_secs=window_secs)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": image_blocks + [{"type": "text", "text": text}],
        },
    ]


def _parse_xml_tool_calls(text: str) -> dict[str, str]:
    """Parse XML-style tool calls emitted by some models as plain text.

    Handles two formats:
    1. <tool_call>{"name": "...", "arguments": {...}}</tool_call>
    2. <tool_call><function=name><parameter=key>value</parameter></function></tool_call>
    """
    result: dict[str, str] = {}
    for block in re.findall(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL):
        block = block.strip()
        # Format 1: JSON body
        try:
            obj = json.loads(block)
            name = obj.get("name", "")
            args = obj.get("arguments", obj.get("parameters", {}))
            if name == "press_button":
                result["press_button"] = args.get("button", "none")
            continue
        except (json.JSONDecodeError, AttributeError):
            pass
        # Format 2: <function=name><parameter=key>value</parameter></function>
        fn_match = re.search(r"<function=(\w+)>(.*?)</function>", block, re.DOTALL)
        if fn_match:
            fn_name = fn_match.group(1)
            fn_body = fn_match.group(2)
            params = {
                k: v.strip()
                for k, v in re.findall(r"<parameter=(\w+)>(.*?)</parameter>", fn_body, re.DOTALL)
            }
            if fn_name == "press_button":
                result["press_button"] = params.get("button", "none")
    return result


def parse_output(output) -> str:
    """Returns the action parsed from tool calls."""
    action = "none"
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
    if action == "none":
        # Fallback: parse XML-style tool calls from raw text
        parsed = _parse_xml_tool_calls(output.text or "")
        button = parsed.get("press_button", "none").strip().lower()
        if button in KEY_LIST_FOR_TRAINING:
            action = button
    return action


def main():

    os.environ.setdefault("VLLM_CACHE_ROOT", "./tmp/vllm")
    from vllm import LLM, SamplingParams
    from emulator.emulator_connection import EmulatorConnection
    from models.util.misc import local_model_map
    from tqdm import tqdm
    import torch

    parser = argparse.ArgumentParser(description="vLLM QwenVL baseline for Pokémon Emerald")
    parser.add_argument("--model", default="Qwen/Qwen3.5-35B-A3B-FP8")
    parser.add_argument("--rom", default=".cache/pokeagent/rom/rom.gba")
    parser.add_argument("--save-state", default=".cache/pokeagent/save_state/truck_start.state")
    parser.add_argument("--steps", type=int, default=72_000)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--video-out", default="./tmp/out", help="Base directory for run output; files go in a timestamped subdirectory")
    parser.add_argument("--save-interval", type=int, default=6000, help="Save emulator state every N steps (0 to disable)")
    parser.add_argument(
        "--local",
        action="store_true",
        help="Write output locally",
    )
    args = parser.parse_args()

    for path in [args.rom, args.save_state]:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Required file not found: {path}")

    run_dir = os.path.join(args.video_out, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    video_path = os.path.join(run_dir, "video")
    print(f"Run output directory: {run_dir}")

    print(f"Connecting to emulator: {args.rom}")
    conn = EmulatorConnection(args.rom)
    conn.load_state_from_file(args.save_state)

    model_path = local_model_map(args.model)

    tensor_parallel_size = torch.cuda.device_count()
    print(f"Loading model: {args.model} (tensor_parallel_size={tensor_parallel_size})")
    llm = LLM(
        model=model_path,
        limit_mm_per_prompt={"image": FRAME_HISTORY_COUNT},
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_tokens,
        tensor_parallel_size=tensor_parallel_size,
        disable_custom_all_reduce=True,
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
    video_segment = 0
    video_path = os.path.join(run_dir, f"video_seg{video_segment:04d}")
    if args.video_out:
        conn.create_video_writer(video_path)
        conn.start_video_writer(video_path)

    log_path = os.path.join(run_dir, "steps.jsonl")
    # Rolling buffer stores every frame; we sample every FRAME_HISTORY_SPACING
    # steps at inference time to build a temporal context window.
    frame_buffer: collections.deque = collections.deque(
        maxlen=FRAME_HISTORY_COUNT * FRAME_HISTORY_SPACING
    )

    print(f"Running for {args.steps} steps")
    with open(log_path, "w") as log_f:
        for step in tqdm(range(args.steps)):
            frame = conn.get_current_frame()
            frame_buffer.append(frame)

            # Sample up to FRAME_HISTORY_COUNT frames spaced FRAME_HISTORY_SPACING apart,
            # most recent last (chronological order).
            buf_len = len(frame_buffer)
            indices = [
                buf_len - 1 - i * FRAME_HISTORY_SPACING
                for i in range(FRAME_HISTORY_COUNT)
                if buf_len - 1 - i * FRAME_HISTORY_SPACING >= 0
            ]
            context_frames = [frame_buffer[idx] for idx in reversed(indices)]

            messages = build_messages(context_frames)
            outputs = llm.chat(
                messages,
                sampling_params=sampling_params,
                tools=TOOLS,
                chat_template_kwargs={"tool_choice": "required", "enable_thinking": False},
                use_tqdm=False,
            )
            output = outputs[0].outputs[0]
            action = parse_output(output)

            conn.set_key(action)
            conn.run_frames(10)
            conn.set_key("none")
            conn.run_frames(50)

            log_f.write(json.dumps({"step": step, "action": action, "raw": output.text}) + "\n")
            log_f.flush()

            if args.save_interval and (step + 1) % args.save_interval == 0:
                state_path = os.path.join(run_dir, f"step{step + 1}.state")
                conn.save_state(state_path)
                print(f"Saved state: {state_path}")

                if args.video_out:
                    conn.release_video_writer(video_path)

                conn.close()
                conn = EmulatorConnection(args.rom)
                conn.load_state_from_file(state_path)

                video_segment += 1
                video_path = os.path.join(run_dir, f"video_seg{video_segment:04d}")
                if args.video_out:
                    conn.create_video_writer(video_path)
                    conn.start_video_writer(video_path)
                print(f"Restarted emulator; new video segment: {video_path}")

    if args.video_out:
        conn.release_video_writer(video_path)
    conn.close()


if __name__ == "__main__":
    main()
