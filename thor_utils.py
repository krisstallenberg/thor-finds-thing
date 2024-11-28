import base64
import numpy as np
import io

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
    