import argparse


def main():
    import torch
    from torchcodec.decoders import VideoDecoder
    from PIL import Image
    import numpy as np
    
    parser = argparse.ArgumentParser(description="Extract milestone frames from video")
    parser.add_argument("--frame-index", type=int, default=0, help="Index of frame to save")
    parser.add_argument("--file-path", type=str, default="tmp/beat_may_and_may_route.mp4", help="Path to video file")

    args = parser.parse_args()
    
    decoder = VideoDecoder(args.file_path, device="cpu")
    
    # Calculate frame range: 15 before, target frame, 16 after (32 frames total)
    start_idx = max(0, args.frame_index - 15)
    end_idx = args.frame_index + 16
    
    # Collect all frames
    frames = []
    for i in range(start_idx, end_idx + 1):
        try:
            frame = decoder[i]
            frames.append(frame)
        except (IndexError, KeyError):
            print(f"Frame {i} out of range, skipping")
            break
    
    # Stack frames into tensor with time dimension: [T, C, H, W]
    frames_tensor = torch.stack(frames, dim=0)
    
    # Save only the requested frame as image
    target_frame = decoder[args.frame_index]
    bounded_frames = None
    if all(v is not None for v in [args.x1, args.y1, args.x2, args.y2]):
        bounded_frames = target_frame.clone()
        bounded_frames[:, args.y1:args.y2, args.x1:args.x2] = 0

    frame_np = target_frame.cpu().numpy().transpose(1, 2, 0)
    image = Image.fromarray(frame_np, mode='RGB')
    image_path = f"./tmp/frame_{args.frame_index}.png"
    image.save(image_path)
    print(f"Saved frame {args.frame_index} as image to {image_path}")
    


if __name__ == "__main__":
    main()