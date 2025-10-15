import torch
from dataset.idm import IDMDataset
from IDM.policy import InverseActionPolicy as IDModel
from policy.policy import CLASS_TO_KEY
from util.data import save_json
import os 
from tqdm import tqdm 

INFERENCE_FPS = 4

SAMPLE_LENGTH = 128
H = 128
W = 128
DEVICE = "cuda:0" 

def load_model(chkpt="/scratch/bsch/pokeagent-speedrun/checkpoints/idm_rnd_data_v2/86cf0631d575026c6e2f1b9a776b4e41cc2eaaf0/4500/rnd_idm_model.pt")-> torch.nn.Module:
    model = IDModel()
    model.load_state_dict(torch.load(chkpt))
    return model

def main(model, video):
    
    log = []

    frames_idx = IDMDataset.process_video(video, INFERENCE_FPS, SAMPLE_LENGTH)

    for sample in tqdm(frames_idx):

        inp = IDMDataset.get_frames(video, sample, H, W, is_val=True)
        inp = torch.unsqueeze(inp, dim=0)
        dummy = {
            "first": torch.zeros((inp.shape[0], 1)).to(DEVICE),
            "state_in": model.initial_state(inp.shape[0])
        }

        # TODO refactor so that this isn't wrapped in a dict
        inp = {"img": inp.to(device=DEVICE)}

        out = model(inp, labels=None, **dummy)
        logits = out.logits
        predictions = torch.argmax(logits, dim=-1).squeeze(dim=0).cpu()

        for i in range(len(sample)):
            log.append({"frame": sample[i], "keys": CLASS_TO_KEY[predictions[i].item()]})    

        dir = os.path.dirname(video)
    save_json(os.path.join(dir, "keys.json"), log)

    
if __name__ == "__main__":
    import sys
    video = sys.argv[1]

    model = load_model()
    model.to(device=DEVICE)
    model.eval()
    main(model, video)