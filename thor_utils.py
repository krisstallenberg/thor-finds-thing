import base64
import numpy as np
import io
from torchvision.transforms.functional import to_tensor
import torch

def encode_image(image):

    # Convert the PIL Image to JPEG format in a byte stream
    byte_stream = io.BytesIO()
    image.save(byte_stream, format="JPEG")
    jpeg_data = byte_stream.getvalue()
    byte_stream.close()

    # Encode the JPEG byte data to base64
    encoded_image = base64.b64encode(jpeg_data).decode('utf-8')
    
    return encoded_image

def get_distance(coord1, coord2):
        """
        Calculate the distance between two coordinates.
        
        Taken from https://github.com/cltl/ma-communicative-robots/blob/2024/emissor_chat/ai2thor_client.py
        
        Returns
        -------
        float
            The distance between the two coordinates.
        """
        distance = np.sqrt((coord2['x'] - coord1['x'])**2
                        + (coord2['y'] - coord1['y'])**2
                        + (coord2['z'] - coord1['z'])**2)
        return distance
    
def closest_objects(objectId, objects, num: int = 1):
    """
    Find the closest object(s) to the given object.
    
    Parameters
    ----------
    objectId : str
        The ID of the object to find the closest object(s) to.
    objects : list
        The list of objects to search through.
    num : int, optional
        The number of closest objects to return. Default is 1.
        
    Returns
    -------
    list of dict
        A list of the closest object(s) to the given object.
    """
    pass

def select_objects(detections_df, target_objects):
    """
    Select the target object that is closest to the center of the context objects.
    """
    # Filter detections
    target_detections = detections_df[detections_df["label"] == target_objects["target"]]
    context_detections = detections_df[detections_df["label"].isin(target_objects["context"])]

    # Handle empty detections
    if target_detections.empty:
        return 'target', None

    if context_detections.empty:
        # Safely return the highest-confidence target if no context objects are found
        return 'target', target_detections.sort_values("confidence", ascending=False).iloc[0].to_dict() if not target_detections.empty else None

    # Calculate distances to context average center
    context_avg_center = context_detections["center_x"].mean()
    target_detections = target_detections.copy()  # Avoid in-place modification
    target_detections["distance_to_context"] = abs(target_detections["center_x"] - context_avg_center)

    # Return the closest target
    closest_target = target_detections.sort_values("distance_to_context").iloc[0].to_dict()
    return 'closest_target', closest_target


def calculate_turn_angle(image_width, object_center_x):
    """Calculate the turn angle based on the horizontal center of the detected object."""
    image_center_x = image_width / 2
    pixel_offset = object_center_x - image_center_x
    degrees_per_pixel = 90 / image_width
    turn_angle = pixel_offset * degrees_per_pixel
    return turn_angle

def expand_box(box, image_width, image_height, padding=30):
    """Expand the bounding box with a margin for better context."""
    x1, y1, x2, y2 = box
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(image_width, x2 + padding)
    y2 = min(image_height, y2 + padding)
    return x1, y1, x2, y2


def calculate_turn_angle(image_width, object_center_x):
    """Calculate the turn angle based on the center of the detected object."""
    image_center_x = image_width / 2
    pixel_offset = object_center_x - image_center_x
    degrees_per_pixel = 90 / image_width
    turn_angle = pixel_offset * degrees_per_pixel
    return turn_angle


def compute_final_angle(objects, image_width):

    if isinstance(objects, dict):  # target object found

        return calculate_turn_angle(image_width, objects["center_x"])
    elif isinstance(objects, list):  # described objects found
        if not objects:  # didnt find the described objects
            print("No described objects identified.")
            return None
        angles = [calculate_turn_angle(image_width, obj["center_x"]) for obj in objects]

        return sum(angles) / len(angles)
    else:  # none found
        print("No target or described objects identified.")
        return None    


