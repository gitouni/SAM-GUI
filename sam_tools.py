from segment_anything import sam_model_registry, SamPredictor
import numpy as np

def get_model(model_name:str, checkpoint_path:str, device:str) -> SamPredictor:
    sam = sam_model_registry[model_name](checkpoint=checkpoint_path).to(device)
    return SamPredictor(sam)

def mask_set_image(model:SamPredictor,image:np.ndarray,image_format="RGB") -> None:
    model.set_image(image=image,image_format=image_format)

def mask_predict(model:SamPredictor, input_points:np.ndarray, point_labels:np.ndarray, input_box:np.ndarray,
                 mask_input:np.ndarray, multimask_output:True):
    mask, scores, _ = model.predict(
        point_coords=input_points,
        point_labels=point_labels,
        box=input_box,
        mask_input=mask_input,
        multimask_output=multimask_output
    )
    return mask, scores