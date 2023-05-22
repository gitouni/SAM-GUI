from segment_anything import sam_model_registry, SamPredictor
import numpy as np

def get_model(model_name:str, checkpoint_path:str, device:str) -> SamPredictor:
    sam = sam_model_registry[model_name](checkpoint=checkpoint_path).to(device)
    return SamPredictor(sam)
def mask_predict(model:SamPredictor, input_points:np.ndarray, point_labels:np.ndarray, input_box:np.ndarray,
                 mask_input:np.ndarray, multimask_output:True):
    mask, scores, logits = model.predict(
        point_coords=input_points,
        point_labels=point_labels,
        box=input_box,
        mask_input=mask_input,
        multimask_output=multimask_output
    )
    return mask, scores, logits

def coord_tran(coord:list,kx:float,ky:float):
    if abs(kx-1) < 1e-4 and abs(ky-1) < 1e-4:
        return coord
    new_coord = []
    assert(len(coord) == 2 or len(coord) == 4), "coord size must be 2 or 4!"
    if len(coord) == 2:
        new_coord.append(coord[0] * kx)
        new_coord.append(coord[1] * ky)
    else:
        new_coord.append(coord[0] * kx)
        new_coord.append(coord[1] * ky)
        new_coord.append(coord[2] * kx)
        new_coord.append(coord[3] * ky)
    return new_coord

def reverse_size(size:list):
    assert(len(size) == 2), "size must has 2 elements"
    return size[1], size[0]