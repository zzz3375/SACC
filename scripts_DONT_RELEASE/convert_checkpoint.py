import argparse
import os
import torch

import sys
sys.path.append("..")
from sam import build_sam

parser = argparse.ArgumentParser(
    description="Map a checkpoint to the new code."
)

parser.add_argument("--input", type=str, required=True, help="Path to input.")
parser.add_argument("--output", type=str, required=True, help="Path to save to.")


def main(args):
    if os.path.exists(args.output):
        raise FileExistsError

    with open(args.input, "rb") as f:
        state_dict = torch.load(f, map_location='cpu')

    # Drop training state
    if 'model' in state_dict:
        print("Dropping training state...")
        state_dict = state_dict['model']

    key_mapping = {}
    output_embed_split = {}
    for k, v in state_dict.items():
        if "interactive_module.output_embed" in k:
            output_embed_split = {
                "mask_decoder.iou_token.weight" : v[:1,:],
                "mask_decoder.mask_tokens.weight" : v[1:,:],
            }
            continue
        if "backbone.net" in k:
            new_k = k.replace("backbone.net", "image_encoder")
        elif "backbone.simfp_4" in k:
            pieces = k.split(".")
            layer = int(pieces[2])
            new_layer = layer * 2 + int(pieces[3] == "norm")
            new_k = "image_encoder.neck." + str(new_layer) + "." + pieces[-1]
        elif "interactive_module.pred_downscaling" in k:
            new_k = k.replace("interactive_module.pred_downscaling", "prompt_encoder.mask_downscaling")
        elif "interactive_module.no_pred_embed" in k:
            new_k = k.replace("interactive_module.no_pred_embed", "prompt_encoder.no_mask_embed")
        elif "interactive_module.point_embeddings" in k:
            new_k = k.replace("interactive_module", "prompt_encoder")
        elif "interactive_module.not_a_point_embed" in k:
            new_k = k.replace("interactive_module", "prompt_encoder")
        elif "interactive_module.pe_layer.positional_encoding_gaussian_matrix" in k:
            new_k = k.replace("interactive_module", "prompt_encoder")
        elif "interactive_module" in k:
            new_k = k.replace("interactive_module", "mask_decoder")
        else:
            new_k = k
        key_mapping[k] = new_k

    new_state_dict = {}
    print("Performing the following maps:")
    for k, new_k in key_mapping.items():
        print(k, "-->", new_k)
        new_state_dict[new_k] = state_dict[k]
    if output_embed_split:
        print("Splitting interactive_module.output_embed.weight")
        for k, v in output_embed_split.items():
            print(f"--> {k} has {v.shape[0]} tokens.")
            new_state_dict[k] = v

    print("Checking that keys match to SAM.")
    sam = build_sam()
    sam_state_dict = sam.state_dict()
    for k, v in sam_state_dict.items():
        assert k in new_state_dict, \
            f"Key {k} missing from generated state_dict!"
        assert v.shape == new_state_dict[k].shape, (
            f"Tensor {k} does not have the necessary shape! "
            f"Target: {v.shape}, generated: {new_state_dict[k].shape}"
        )
    assert set(new_state_dict.keys()) - set(sam_state_dict.keys()) == set(), \
        f"Extra keys in generated state dict: {set(new_state_dict.keys()) - set(sam_state_dict.keys())}"

    print("Saving new model to", args.output)
    with open(args.output, "wb") as g:
        torch.save(new_state_dict, g)

if __name__=="__main__":
    args = parser.parse_args()
    main(args)


