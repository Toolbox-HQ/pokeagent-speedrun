import argparse
import sys
from emulator.emulator_manual_client import run

def main():
    parser = argparse.ArgumentParser(description="Manual mode Pokemon Emerald")
    parser.add_argument("--rom", type=str, default=".cache/lz/rom/lz_rom.gba")
    parser.add_argument("--mp4-path", type=str, default=".cache/lz/manual_output.mp4")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--keys-json-path", type=str, default=".cache/lz/manual_keys.json")
    parser.add_argument("--save-state", type=str, default=None)
    parser.add_argument("--max-steps", type=int, default=0)
    args = parser.parse_args()

    run(
        rom=args.rom,
        mp4_path=args.mp4_path,
        port=args.port,
        manual_mode=True,
        fps=args.fps,
        keys_json_path_local=args.keys_json_path,
        save_state=args.save_state,
        max_steps=args.max_steps,
    )

if __name__ == "__main__":
    sys.exit(main())
