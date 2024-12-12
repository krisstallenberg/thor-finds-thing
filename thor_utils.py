import base64
import numpy as np
import io
from pandas import DataFrame
from descriptions import ObjectMapping

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