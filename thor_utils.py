import base64
import numpy as np
import io
from pandas import DataFrame
from descriptions import ObjectMapping
import math

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



def calculate_turn_and_distance_dot_product(agent_pos, agent_rotation, object_pos):
    """
    Calculate the turn angle and distance between an agent and an object using the dot product formula.
    Args:
        agent_pos (dict): The agent's position with keys 'x', 'y', 'z'.
        agent_rotation (dict): The agent's rotation with keys 'x', 'y', 'z' (pitch, yaw, roll).
        object_pos (dict): The object's position with keys 'x', 'y', 'z'.
    Returns:
        tuple: (turn_angle, distance)
    """
    # Extract x, y positions
    x1, z1 = agent_pos  # Using 'z' for 2D plane (x, z)
    x2, z2 = object_pos

    # Calculate the distance
    distance = math.sqrt((x2 - x1) ** 2 + (z2 - z1) ** 2)

    # Direction vector to the object
    dir_vector = (x2 - x1, z2 - z1)

    # Agent's facing vector based on yaw
    yaw = math.radians(agent_rotation['y'])  # Convert yaw to radians
    agent_facing = (math.cos(yaw), math.sin(yaw))

    # Dot product
    dot_product = agent_facing[0] * dir_vector[0] + agent_facing[1] * dir_vector[1]

    # Magnitude of direction vector
    magnitude_dir = math.sqrt(dir_vector[0] ** 2 + dir_vector[1] ** 2)

    # Avoid division by zero
    if magnitude_dir == 0:
        return 0, 0

    # Calculate the angle
    cos_theta = dot_product / magnitude_dir
    theta = math.acos(cos_theta)  # Angle in radians

    # Use cross product to determine the direction (sign of the angle)
    cross_product = agent_facing[0] * dir_vector[1] - agent_facing[1] * dir_vector[0]
    if cross_product < 0:
        theta = -theta  # Negative angle for clockwise

    return theta, distance


    pass
    
def map_detected_to_visible_objects(detected_objects, visible_objects: list, openai_client) -> dict:
    detected_labels = detected_objects['label'].tolist()
    visible_object_ids = [object['objectId'] for object in visible_objects]
       
    response = openai_client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[
            {
                "role": "system",
                "content": """You are tasked to match a list of user-defined objects to object identifiers in a space. 
The user-defined objects are an open set. 
For example, one user may say 'couch', where another says 'sofa', while referring to the same object. 
The object identifiers consist of a string, with coordinates concatenated to them. For example: 'AlarmClock|-02.08|+00.94|-03.62'.
In your mapping, make sure to preserve the full objectId string, but don't take the coordinates into account when finding a matching user-defined object.

For example, if the user-defined list of objects contains a string 'Couch' and the list of object identifiers contains strings 'Sofa|-02.08|+00.94|-03.62', 'Sofa|-04.32|+01.73|-01.02', map both object identifiers to Couch.
Respond with an object mapping the user-defined object names as keys to lists of object identifiers as values, for example:

[
    {user_defined_name: "Alarm", objectIds: ["AlarmClock|-02.08|+00.94|-03.62"]},
    {user_defined_name: "Couch", objectIds: ["Sofa|-02.08|+00.94|-03.62", "Sofa|-04.32|+01.73|-01.02"]},
    {user_defined_name: "Painting", objectIds: ["Painting|-05.08|+00.74|-03.22"]}
]
"""
},
            {
                "role": "user",
                "content": f"Here are the user-defined object names: {detected_labels}.\nHere are the object identifiers: {visible_object_ids}"
            },
        ],
        response_format=ObjectMapping
    )
    name_mapping = response.choices[0].message.parsed
    name_mapping_dict = {obj.user_defined_name: obj.objectIds for obj in name_mapping.mapping}
        
    return name_mapping_dict

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
