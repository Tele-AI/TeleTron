import os
import re
import torch
import safetensors.torch
from collections import OrderedDict


# model_path 是str或者list[str]
def get_normal_state_dict(model_path: str | list[str]):
    if isinstance(model_path, str):
        if model_path.endswith(".safetensors"):
            return safetensors.torch.load(open(model_path, "rb").read())
        else:
            return torch.load(model_path, map_location="cpu", weights_only=False)
    else:
        assert isinstance(model_path, list)
        state_dict = OrderedDict()
        for path in model_path:
            state_dict.update(get_normal_state_dict(path))
        return state_dict

def update_state_dict(state_dict):
    output_state_dict = OrderedDict()
    replacement_rules = [
        (r'\.k\.','.key.'),
        (r'\.q\.','.query.'),
        (r'\.v\.','.value.'),
        (r'\.o\.','.out_proj.'),
        (r'\.norm_q\.','.norm_query.'),
        (r'\.norm_k\.','.norm_key.'),
        (r'\.k_img\.','.img_key.'),
        (r'\.v_img\.','.img_value.'),
        (r'\.norm_k_img\.','.norm_image_key.'),
        (r'^patch_embedding\.','patch_emb.'),
        (r'^time_embedding\.','time_emb.'),
        (r'^text_embedding\.','text_emb.'),
        (r'^time_projection\.','time_proj.'),
    ]
    for key, value in state_dict.items():
        for old, new in replacement_rules:
            key = re.sub(old, new, key)
        output_state_dict[key] = value
    return output_state_dict

def save_teletron_release(state_dict, checkpoint_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)
    latest_file = os.path.join(checkpoint_dir, "latest_checkpointed_iteration.txt")
    with open(latest_file, "w") as f:
        f.write("release")
    release_dir = os.path.join(checkpoint_dir, "release")
    os.makedirs(release_dir, exist_ok=True)
    checkpoint_dir = os.path.join(release_dir, "mp_rank_00")
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_name = "model_optim_rng.pt"
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
    output_state_dict = OrderedDict({
            "args": None,  
            "checkpoint_version": 3.0,
            "model":  {k: v for k, v in state_dict.items()},
        })
    torch.save(output_state_dict, checkpoint_path)


if __name__ == "__main__":
    update_key_from_wan_to_teleai = True
    model_path = [f"/nvfile-heatstorage/model_zoo/modelscope/Wan2.2-TI2V-5B/diffusion_pytorch_model-0000{i}-of-00003.safetensors" for i in range(1, 4)]
    target_path = "/nvfile-heatstorage/AIGC_H100/shanggonghu/project/text2video/Teletron_latest/checkpoint/sgh_5B/"
    state_dict = get_normal_state_dict(model_path)
    print(f"successfully load model from {model_path}")
    if update_key_from_wan_to_teleai:
        state_dict = update_state_dict(state_dict)
        print(f"successfully update state_dict")
    save_teletron_release(state_dict, target_path)
    print(f"successfully save model to {target_path}")