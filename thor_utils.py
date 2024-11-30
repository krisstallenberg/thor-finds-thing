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
    pass

def detect_objects(image, confidence_threshold=0.75):
    """Detect objects using Faster R-CNN."""
    image_width, image_height = image.size
    tensor_image = F.to_tensor(image).unsqueeze(0)

    # Run Faster R-CNN
    # frcnn_model.eval()
    with torch.no_grad():
        detections = self._frcnn_model(tensor_image)[0]

    # Filter detections by confidence
    valid_detections = [
        {
            "x1": box[0].item(),
            "y1": box[1].item(),
            "x2": box[2].item(),
            "y2": box[3].item(),
            "confidence": score.item(),
        }
        for box, score in zip(detections["boxes"], detections["scores"])
        if score.item() >= confidence_threshold
    ]
    return valid_detections, image_width, image_height



def classify_objects(detections, image, target_objects, padding=30):
    """Classify detected objects using CLIP."""
    image_width, image_height = image.size
    detections_with_labels = []

    for det in detections:
        # Expand bounding box
        x1, y1, x2, y2 = expand_box(
            (det["x1"], det["y1"], det["x2"], det["y2"]),
            image_width,
            image_height,
            padding,
        )
        cropped_image = image.crop((x1, y1, x2, y2))

        # Preprocess for CLIP
        inputs = self._clip_processor(
            text=[target_objects["target"]] + target_objects["described"],
            images=cropped_image,
            return_tensors="pt",
            padding=True,
        )

        # Run CLIP classification
        with torch.no_grad():
            outputs = self._clip_model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1).squeeze(0)

        # Assign label
        labels = [target_objects["target"]] + target_objects["described"]
        max_prob_idx = torch.argmax(probs).item()
        detected_label = labels[max_prob_idx]
        confidence = probs[max_prob_idx].item()

        # Append result
        object_center_x = (x1 + x2) / 2
        detections_with_labels.append({
            "label": detected_label,
            "confidence": confidence,
            "center_x": object_center_x,
            "box": (x1, y1, x2, y2)
        })

    return detections_with_labels



def select_objects(detections_df, target_objects):
    """Select the target and described objects."""
    # Handle target object
    target_detections = detections_df[detections_df["label"] == target_objects["target"]]
    described_detections = detections_df[detections_df["label"].isin(target_objects["described"])]

    # Select the closest target
    if not target_detections.empty:
        if not described_detections.empty:
            described_avg_center = described_detections["center_x"].mean()
            target_detections["distance"] = abs(target_detections["center_x"] - described_avg_center)
            closest_target = target_detections.sort_values("distance").iloc[0].to_dict()  # Convert Series to dict

            return "target", closest_target

    # Handle described objects
    described_objects = target_objects["described"]
    final_described = []
    for obj in described_objects:
        obj_detections = detections_df[detections_df["label"] == obj]
        num_required = described_objects.count(obj)

        if not obj_detections.empty:
            if len(obj_detections) >= num_required:
                obj_detections = obj_detections.sort_values("confidence", ascending=False).head(num_required)
            final_described.extend(obj_detections.to_dict(orient="records"))  # Convert to list of dicts


    return "described", final_described


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


