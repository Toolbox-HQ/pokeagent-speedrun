import argparse
import os
import shutil

from models.dataset.idm import IDMDataset


def main():
    parser = argparse.ArgumentParser(
        description="Copy IDM data to a new folder and filter out actions that do not change the screen."
    )
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--buffer_size", type=int, default=60)
    args = parser.parse_args()

    assert os.path.isdir(args.input_dir), f"input_dir does not exist: {args.input_dir}"
    assert not os.path.exists(args.output_dir) or not os.listdir(args.output_dir), \
        f"output_dir already exists and is not empty: {args.output_dir}"

    print(f"Copying {args.input_dir} -> {args.output_dir} ...")
    shutil.copytree(args.input_dir, args.output_dir)

    print(f"Applying action filter on {args.output_dir} ...")
    IDMDataset(
        data_path=args.output_dir,
        apply_filter=True,
        buffer_size=args.buffer_size,
    )

    print(f"Done. Filtered data written to {args.output_dir}")


if __name__ == "__main__":
    main()
