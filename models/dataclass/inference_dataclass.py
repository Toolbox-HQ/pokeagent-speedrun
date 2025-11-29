from dataclasses import dataclass, field
from typing import Optional

@dataclass
class InferenceArguments:
    
    # be careful for overlapped keys in other classes
    infernece_architecture: Optional[str] = field(default=None)
    
    model_checkpoint: Optional[str] = field(default=None)

    agent_steps: Optional[int] = field(default=None)
    
    save_state: Optional[str] = field(default=None)
    
    sampling_strategy: Optional[str] = field(default=None)
    
    temperature: Optional[float] = field(default=1)

    actions_per_seconds: Optional[int] = field(default=1)

    inference_save_path: Optional[str] = field(default=None) 

    agent_fps: Optional[int] = field(default=None)

    context_length: Optional[int] = field(default=None)
