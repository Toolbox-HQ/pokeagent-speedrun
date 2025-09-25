import sys
import pickle
from argparse import ArgumentParser
import torch as th
from lib.torch_util import default_device_type, set_default_torch_device
from policy import InverseActionPolicy
from agent import resize_image, AGENT_RESOLUTION
import numpy as np
import cv2

net_kwargs = {   
        'attention_heads': 32,
        'attention_mask_style': 'none',
        'attention_memory_size': 128,
        'conv3d_params': {'inchan': 3, 'kernel_size': [5, 1, 1], 'outchan': 128, 'padding': [2, 0, 0]},
        'hidsize': 4096, 'img_shape': [128, 128, 128],
        'impala_kwargs': {'post_pool_groups': 1},
        'impala_width': 16,
        'init_norm_kwargs': {'batch_norm': False, 'group_norm_groups': 1},
        'n_recurrence_layers': 2,
        'only_img_input': True,
        'pointwise_ratio': 4,
        'pointwise_use_activation': False,
        'recurrence_is_residual': True,
        'recurrence_type': 'transformer',
        'single_output': True,
        'timesteps': 128,
        'use_pointwise_layer': True,
        'use_pre_lstm_ln': False
    }


class IDMAgent:

    def __init__(self, idm_net_kwargs, device=None):
        if device is None:
            device = default_device_type()
        self.device = th.device(device)
        set_default_torch_device(self.device)
        self.policy = InverseActionPolicy(idm_net_kwargs).to(device)

        self.hidden_state = self.policy.initial_state(1)
    
    def load_weights(self, path):
        """Load model weights from a path, and reset hidden state"""
        self.policy.load_state_dict(th.load(path, map_location=self.device), strict=False)
        self.reset()

    def reset(self):
        """Reset agent to initial state (i.e., reset hidden state)"""
        self.hidden_state = self.policy.initial_state(1)

    def _video_obs_to_agent(self, video_frames):
        imgs = [resize_image(frame, AGENT_RESOLUTION) for frame in video_frames]
        # Add time and batch dim
        imgs = np.stack(imgs)[None]
        agent_input = {"img": th.from_numpy(imgs).to(self.device)}
        return agent_input

    def predict_actions(self, video_frames):
        """
        Predict actions for a sequence of frames.

        `video_frames` should be of shape (N, H, W, C).
        Returns MineRL action dict, where each action head
        has shape (N, ...).

        Agent's hidden state is tracked internally. To reset it,
        call `reset()`.
        """
        agent_input = self._video_obs_to_agent(video_frames)
        # The "first" argument could be used to reset tell episode
        # boundaries, but we are only using this for predicting (for now),
        # so we do not hassle with it yet.
        dummy_first = th.zeros((video_frames.shape[0], 1)).to(self.device)
        predicted_actions, self.hidden_state, _ = self.policy.predict(
            agent_input, first=dummy_first, state_in=self.hidden_state,
            deterministic=True
        )

        return predicted_actions

def main(video_path, n_batches, n_frames):
    cap = cv2.VideoCapture(video_path)
    agent = IDMAgent(idm_net_kwargs=net_kwargs)

    recorded_actions = []
    for _ in range(n_batches):
        th.cuda.empty_cache()
        frames = []
        for _ in range(n_frames):
            ret, frame = cap.read()
            if not ret:
                break
            # BGR -> RGB
            frames.append(frame[..., ::-1])
        frames = np.stack(frames)
        print("=== Predicting actions ===")
        recorded_actions.append(agent.predict_actions(frames))
    print(recorded_actions)    

if __name__ == "__main__":
    parser = ArgumentParser("Run IDM on Pokemon emerald recordings.")

    #parser.add_argument("--weights", type=str, required=True, help="Path to the '.weights' file to be loaded.")
    #parser.add_argument("--model", type=str, required=True, help="Path to the '.model' file to be loaded.")
    parser.add_argument("--video-path", type=str, required=True, help="Path to a .mp4 file (Pokemon emerald recording).")
    parser.add_argument("--n-frames", type=int, default=128, help="Number of frames to process at a time.")
    parser.add_argument("--n-batches", type=int, default=10, help="Number of batches (n-frames) to process for visualization.")
    args = parser.parse_args()
    main(args.video_path, args.n_batches, args.n_frames)