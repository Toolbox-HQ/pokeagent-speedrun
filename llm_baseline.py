import argparse
import base64
from io import BytesIO
from emulator.keys import KEY_LIST_FOR_TRAINING
import json
import os
import re

SYSTEM_PROMPT = """\
You are playing Pokémon Emerald on a Game Boy Advance emulator.
You will be shown the current game screen and must choose the single best action to make progress in the game.
You must call one tool every step: press_button or update_memory.

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
            elif name == "update_memory":
                result["update_memory"] = args.get("memory", "")
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
            elif fn_name == "update_memory":
                result["update_memory"] = params.get("memory", "")
    return result


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
    if action == "none" or memory is None:
        # Fallback: parse XML-style tool calls from raw text
        parsed = _parse_xml_tool_calls(output.text or "")
        if action == "none":
            button = parsed.get("press_button", "none").strip().lower()
            if button in KEY_LIST_FOR_TRAINING:
                action = button
        if memory is None:
            memory = parsed.get("update_memory")
    return action, memory or ""


def main():

    os.environ.setdefault("VLLM_CACHE_ROOT", "./tmp/vllm")
    from vllm import LLM, SamplingParams
    from emulator.emulator_connection import EmulatorConnection
    from models.util.misc import local_model_map
    import tqdm
    import torch

    parser = argparse.ArgumentParser(description="vLLM QwenVL baseline for Pokémon Emerald")
    parser.add_argument("--model", default="Qwen/Qwen3.5-35B-A3B-FP8")
    parser.add_argument("--rom", default=".cache/pokeagent/rom/rom.gba")
    parser.add_argument("--save-state", default=".cache/pokeagent/save_state/truck_start.state")
    parser.add_argument("--steps", type=int, default=5_000)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--video-out", default="./tmp/out", help="Path prefix for output video (omit extension)")
    parser.add_argument("--save-interval", type=int, default=1000, help="Save emulator state every N steps (0 to disable)")
    parser.add_argument(
        "--local",
        action="store_true",
        help="Write output locally",
    )
    args = parser.parse_args()

    for path in [args.rom, args.save_state]:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Required file not found: {path}")

    print(f"Connecting to emulator: {args.rom}")
    conn = EmulatorConnection(args.rom)
    conn.load_state_from_file(args.save_state)

    model_path = local_model_map(args.model) if args.local else args.model

    tensor_parallel_size = torch.cuda.device_count()
    print(f"Loading model: {args.model} (tensor_parallel_size={tensor_parallel_size})")
    llm = LLM(
        model=model_path,
        limit_mm_per_prompt={"image": 1},
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_tokens,
        tensor_parallel_size=tensor_parallel_size,
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

    print(f"Running for {args.steps} steps")
    for step in tqdm(range(args.steps)):
        print("[GET FRAME FROM EMULATOR]")
        frame = conn.get_current_frame()
        messages = build_messages(frame, memory)
        print("[DO CHAT]")
        outputs = llm.chat(
            messages,
            sampling_params=sampling_params,
            tools=TOOLS,
            chat_template_kwargs={"tool_choice": "required"},
        )
        output = outputs[0].outputs[0]
        print(f"[RAW CHAT] text={repr(output.text)}")
        action, memory = parse_output(output)

        conn.set_key(action)
        conn.run_frames(1)
        conn.set_key("none")
        conn.run_frames(59)

        print(f"Step {step}/{args.steps} | action={action} | memory={repr(memory)}")

        if args.save_interval and step > 0 and step % args.save_interval == 0:
            save_path = os.path.join(os.path.dirname(args.video_out) or ".", f"step{step}.state")
            conn.save_state(save_path)
            print(f"Saved state: {save_path}")

    if args.video_out:
        conn.release_video_writer(args.video_out)
    conn.close()


if __name__ == "__main__":
    main()
