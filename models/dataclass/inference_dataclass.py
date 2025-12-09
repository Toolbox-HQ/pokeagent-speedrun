from dataclasses import dataclass, field
from typing import Optional

@dataclass
class InferenceArguments:
    
    # be careful for overlapped keys in other classes
    inference_architecture: Optional[str] = field(default=None)
    
    model_checkpoint: Optional[str] = field(default=None)

    agent_steps: Optional[int] = field(default=None)
    
    save_state: Optional[str] = field(default=None)
    
    sampling_strategy: Optional[str] = field(default=None)
    
    temperature: Optional[float] = field(default=1)

    actions_per_second: Optional[int] = field(default=1)

    inference_save_path: Optional[str] = field(default=None) 

    agent_fps: Optional[int] = field(default=None)

    context_length: Optional[int] = field(default=None)

    online: Optional[bool] = field(default=False)

    rom_path: Optional[str] = field(default=None)

    idm_data_sample_interval: Optional[int] = field(default=None)

    idm_data_sample_steps: Optional[int] = field(default=None)

    bootstrap_interval: Optional[int] = field(default=None)

    match_length: Optional[int] = field(default=None)

    retrieved_videos: Optional[int] = field(default=None)

    max_vid_len: float = field(default=None)

    def __iter__(self):
        return iter(vars(self).values())
