from ai2thor.controller import Controller
from PIL import Image
from llama_index.llms.openai import OpenAI as OpenAILlamaIndex
from llama_index.llms.ollama import Ollama as OllamaLlamaIndex
from descriptions import InitialDescription, ViewDescription
from leolani_client import Action
from openai import OpenAI
# from torch.nn import functional as F
from fastapi.encoders import jsonable_encoder as F

import random
import base64
import torchvision
from transformers import CLIPProcessor, CLIPModel
import json
import time
from thor_utils import ( 
                        encode_image, 
                        get_distance,
                        closest_objects,
                        detect_objects,
                        classify_objects,
                        select_objects,
                        calculate_turn_angle,
                        expand_box,
                        calculate_turn_angle,
                        compute_final_angle
			)
# Constants
VISIBILITY_DISTANCE = 1.5
SCENE = "FloorPlan212"

class AI2ThorClient: 
    """
    An AI2Thor instance with methods wrapping its controller.
    """

    def __init__(self, leolaniClient, chat_mode):
        self._controller = Controller(
            agentMode="default",
            visibilityDistance=VISIBILITY_DISTANCE,
            scene=SCENE,

            # step sizes
            gridSize=0.25,
            snapToGrid=True,
            rotateStepDegrees=90,

            # image modalities
            renderDepthImage=False,
            renderInstanceSegmentation=False,

            # camera properties
            width=512,
            height=512,
            fieldOfView=90
            )
        self._metadata = []
        self.descriptions = []
        self.unstructured_descriptions = []
        self.context_objects = ['sofa', 'painting', 'table']
        self.target_object = 'lamp'
        self.global_dict = {}
        self.leolaniClient = leolaniClient
        self._llm_ollama = OllamaLlamaIndex(model="llama3.2", request_timeout=120.0)
        self._llm_openai = OpenAILlamaIndex(model="gpt-4o-2024-08-06")
        self._llm_openai_multimodal = OpenAI()
        self._clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()
        self._frcnn_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True).eval()
        self._clip_processor = clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def describe_view_from_image(self):
        """
        Describes the current view using an image-to-text model.

        Returns
        -------
        str
            A string describing the current view.
        """
        encoded_image = encode_image(self._get_image())

        response = self._llm_openai_multimodal.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Imagine this is your point-of-view. Describe what you see in this virtual environment. Write from the first perspective so start your message with 'I'. First, describe the objects, their colors, and their positions. Don't introduce your description. Start describing directly e.g. 'I currently see a <object> on a <surface> ...'. Be objective in your description! Finally describe the room type: it's either a living room, kitchen, bedroom, or bedroom. It can't be anything else. If you can't infer the room type, just say so.",
                        },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url":  f"data:image/jpeg;base64,{encoded_image}"
                            },
                        },
                    ],
                },
            ],
        )
        
        self.descriptions.append(response.choices[0].message.content)
        return response.choices[0].message.content


    def describe_view_from_image_structured(self):
        """
        Describes the current view using an image-to-text model with structure.
        
        Returns:
        -------
        ViewDescription
            A structured description of the current view.
        """    

        encoded_image = encode_image(self._get_image())
        
        response = self._llm_openai_multimodal.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Imagine this is your point-of-view. Describe what you see in this virtual environment. Write from the first perspective. Describe the objects, their colors, and their positions. Be objective in your description! Describe the room type: it's either a living room, kitchen, bedroom, or bedroom. It can't be anything else.",
                            },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url":  f"data:image/jpeg;base64,{encoded_image}"
                                },
                            },
                        ],
                    },
                ],
            response_format=ViewDescription,
            )
        
        self.descriptions.append(response.choices[0].message.parsed)
        return response.choices[0].message.parsed

    def infer_room_type(self, description: str) -> str:
        """
        Infers the room type the agent is in.

        Inference is based on:
        - The image-to-text description of the view.
        - The objects in the metadata.
        - The AI2Thor object types mapping (https://ai2thor.allenai.org/ithor/documentation/objects/object-types).

        Returns
        -------
        Returns a string representing the likely room type.
        """
        pass 
    
    def parse_unstructured_description(self, description: str):
        """
        Parse an unstructured description into structured data.

        Parameters
        ----------
        description : str
            The unstructured description to parse.
            
        Returns
        -------
        PydanticModel
            An instance of the given Pydantic model populated with the parsed data.
        """

        response = self._llm_openai_multimodal.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            messages=[
                {"role": "system", "content": """Your task is to turn a user's description of an object, its context and the room type into a structured response. 
                 When information is missing from the user's description, do not make up parts of the description, go ONLY off of the user's description. 
                 Only deviate from this rule when positions of objects in context are obvious, such as a floor (which is always below the target object) and a ceiling (which is above)."""},
                {"role": "user", "content": description}
            ],
            response_format=InitialDescription,
        )

        self.structured_initial_description = response.choices[0].message.parsed

    def _find_objects_and_angles(self, image, target_objects):
        """Main workflow to find objects and compute turn angles."""

        # Detection
        detections, image_width, image_height = detect_objects(image)

        # Classification
        detections_with_labels = classify_objects(detections, image, target_objects)

        detections_df = pd.DataFrame(detections_with_labels)

        # Object Selection
        obj_type, selected_objects = select_objects(detections_df, target_objects)

        # Angle Calculation
        turn_angle = compute_final_angle(selected_objects, image_width)

        return turn_angle, detections_df, selected_objects

    def find_and_go_to_object(self):
        """
        Locate the target object and move toward it.
        If the target is not found initially, go to the middle of the room and retry.
        If the object is still not found, return None.
        
        Args:
            targets (dict): A dictionary containing:
                - 'target_object': {'name': str} (Target object name).
                - 'context': [{'name': str}] (Optional described objects in the picture).
        
        Returns:
            tuple: (object_id, list) where:
                - object_id: The ID of the object the agent approached, or None if unsuccessful.
                - list: A list of log messages generated during the process.
        """
        # targets=self.descriptions
        logs = []  # List to collect log messages
        # target_label = targets['target_object']['name']
        # context_objects = [item['name'] for item in targets.get('context', [])]
        target_label=self.target_object
        context_objects=self.context_objects
    
        # Step 1: Attempt to find and go to the target object
        object_id, logs = self._attempt_to_find_and_go_to_target(target_label, context_objects, logs)
        if object_id is None:
            logs.append(f"Target '{target_label}' not found. Going to the middle of the room.")
            
            # Step 2: Go to the middle of the room
            thor._teleport(position=thor._find_nearest_center_of_room())
            logs.append("Teleported to the middle of the room.")
        
            # Step 3: Retry finding and going to the target object
            object_id, logs = self._attempt_to_find_and_go_to_target(target_label, context_objects, logs)
            if object_id is None:
                logs.append(f"Target '{target_label}' still not found. Returning None.")
                return None, logs  # Target not found even after retrying
    
        logs.append(f"Target '{target_label}' successfully reached. Object ID: {object_id}")
        return object_id, logs  # Return the object ID and logs



    def _attempt_to_find_and_go_to_target(self, target_label: str, context_objects: list, logs: list) -> tuple:
        """
        Attempt to locate the target object and move toward it.
        If the agent cannot step, teleport closer to the target.
    
        Args:
            target_label (str): Name of the target object.
            context_objects (list): List of described object names.
            logs (list): A list to collect log messages.
    
        Returns:
            tuple: (object_id, list) where:
                - object_id: The ID of the object the agent approached, or None if unsuccessful.
                - list: A list of log messages.
        """
        turn_angle = None  
        count_of_turns = 0  
    
        while turn_angle is None:
            image = self._get_image()  # Update the image after each rotation

            # Define target and described objects
            target_objects = {'target': target, 'context': context}

            # Run the workflow to find objects and calculate turn angles
            turn_angle, detections_df, objects_selected = self._find_objects_and_angles(
                image=image,
                target_objects={'target': target_label,
                                'context': context_objects}
            )
            self.rotate(direction='left')  # Rotate to search for objects
            logs.append(f"Rotated left to search for '{target_label}'.")
            count_of_turns += 1
    
            if count_of_turns == 4:  # If the object is not found after 4 rotations
                logs.append(f"Target '{target_label}' not found after 4 rotations.")
                return None, logs  # Target not found
    
        # Map target label to visible objects
        visible_objects = self._find_objects_in_sight()
        matched_objects = self._map_target_to_visible_objects(target_label, visible_objects)
    
        if not matched_objects:
            logs.append(f"No semantically matching objects found for target '{target_label}'.")
            return None, logs
    
        # Select the best object based on proximity to described objects
        # selected_object = self._select_best_object(matched_objects, context_objects, visible_objects)
        logs.append(f"Matched target '{target_label}' to object: {selected_object}.")
        for item in matched_objects:
            for obj in visible_objects:
                if obj['id'] == item:
                    selected_object=obj
        
        
        # Rotate to align with the selected object
        if turn_angle > 0:
            self._rotate(direction='right', degrees=abs(turn_angle))
            logs.append(f"Rotated right by {abs(turn_angle)} degrees to align with the target.")
        else:
            self._rotate(direction='left', degrees=abs(turn_angle))
            logs.append(f"Rotated left by {abs(turn_angle)} degrees to align with the target.")
    
        # Step toward the target object
        max_teleports = 10  # Prevent infinite retries
        teleport_count = 0
    
        while teleport_count <= max_teleports:
            while True:
                agent_position = self.metadata["position"]
                target_position = selected_object["position"]
    
                # Check if the target is within 3 meters
                distance = get_distance(agent_position, target_position)
                if distance <= 3:
                    logs.append(f"Target '{target_label}' is within {distance:.2f} meters. Successfully reached.")
                    return selected_object["id"], logs  # Return the ID of the target object
    
                # Try stepping toward the target
                if not self._step():
                    logs.append("Step failed. Calculating closest teleportable position.")
                    break  # Exit to handle teleportation
    
            # Handle teleportation if stepping fails
            teleport_position = self._calculate_closest_teleportable_position(agent_position, target_position)
            if teleport_position is None:
                logs.append("No suitable teleportable position found.")
                return None, logs
    
            self._teleport(to=teleport_position)
            teleport_count += 1
            logs.append(f"Teleported to {teleport_position}. Resuming movement.")
    
        logs.append("Max teleports reached. Could not reach the target.")
        return None, logs

        objects=[]
        if type(objects_selected)==dict:
            objects.append(objects_selected['label'])
        else:
            for item in objects_selected:
                objects.append(item['label'])
        # approach the target object
        while True:
            visible_objects = self._find_objects_in_sight(object_type=[objects[0]])
            if target in visible_objects:  
                if target in objects[0]:
                    return True, 'target'    # Target object is in sight
                else:
                    return True, 'context'
            if not self._step():
                print("Step failed")
                return False
                               # This part needs more work, if it cant step is gets stuck

    def _map_target_to_visible_objects(self, target_label: str, visible_objects: list) -> list:
        """
        Map the target label to the most semantically similar object(s) in visible objects using an LLM.
        Return all objects of the same type if there are multiple matches.
    
        Args:
            target_label (str): The target label to map.
            visible_objects (list): List of visible objects with their IDs and types.
    
        Returns:
            list: A list of tuples containing the IDs and positions of the matched objects.
        """
        # Extract object types and positions from visible objects
        object_data = [{"id": obj["id"], "type": obj["label"], "position": obj["position"]} for obj in visible_objects]
        object_descriptions = [f"Object ID: {obj['id']}, Type: {obj['type']}" for obj in object_data]
    
        # Compose the LLM prompt
        prompt = (
            f"Match the target label '{target_label}' to the objects in the list below based on semantic similarity. "
            f"If multiple objects are of the same type and match, return all their IDs. The list of objects is:\n"
            + "\n".join(object_descriptions)
        )
    
        # Query the LLM
        response = self._llm_openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
    
        # Extract the LLM response
        matched_ids = response.choices[0].message.content.strip().split(",")
        matched_ids = [id.strip() for id in matched_ids if id.strip()]  # Clean up the list
    
        # Validate and collect matched objects
        matched_objects = [
            {"id": obj["id"], "position": obj["position"], "type": obj["type"]}
            for obj in object_data if obj["id"] in matched_ids
        ]
    
        if not matched_objects:
            print(f"No semantically matching objects found for target '{target_label}'.")
            return []
    
        # If multiple objects of the same type exist, include all of them
        types = [obj["type"] for obj in matched_objects]
        if len(set(types)) == 1:  # All objects are of the same type
            matched_objects = [
                obj for obj in object_data if obj["type"] == types[0]
            ]
    
        # Calculate positions relative to the agent's current direction
        agent_position = self.metadata["position"]
        agent_rotation = self.metadata["rotation"]
        for obj in matched_objects:
            obj["relative_position"] = self._calculate_relative_position(
                agent_position, agent_rotation, obj["position"]
            )
    
        print(f"Matched objects for target '{target_label}': {matched_objects}")
        return matched_objects
    def _calculate_relative_position(self, agent_position: dict, agent_rotation: dict, object_position: dict) -> dict:
        """
        Calculate the relative position of an object with respect to the agent's direction.
    
        Args:
            agent_position (dict): The agent's current position (x, y, z).
            agent_rotation (dict): The agent's current rotation (yaw, pitch, roll).
            object_position (dict): The object's position (x, y, z).
    
        Returns:
            dict: A dictionary with relative position information:
                  - distance (float): The Euclidean distance between the agent and the object.
                  - angle (float): The relative angle of the object to the agent's forward direction.
        """
        import math
    
        # Calculate the vector from the agent to the object
        dx = object_position["x"] - agent_position["x"]
        dz = object_position["z"] - agent_position["z"]
    
        # Calculate the Euclidean distance
        distance = (dx**2 + dz**2) ** 0.5
    
        # Calculate the angle relative to the agent's forward direction
        
        agent_yaw = agent_rotation["yaw"]  # Agent's forward direction (in degrees)
        object_angle = math.degrees(math.atan2(dz, dx))  # Object's angle relative to the origin
        relative_angle = (object_angle - agent_yaw + 360) % 360  # Normalize to [0, 360)
    
        # Normalize angle to [-180, 180] for easier directional interpretation
        if relative_angle > 180:
            relative_angle -= 360
    
        return {"distance": distance, "angle": relative_angle}
    
    
    def _calculate_closest_teleportable_position(self, agent_position: dict, target_position: dict) -> dict:
        """
        Calculate the closest teleportable position along the line between the agent and the target.
    
        Args:
            agent_position (dict): Current position of the agent (x, y, z).
            target_position (dict): Position of the target object (x, y, z).
    
        Returns:
            dict or None: The closest teleportable position if found, otherwise None.
        """
        teleportable_positions = self.metadata["teleportable_positions"]
    
        closest_position = None
        min_distance = float('inf')
    
        for pos in teleportable_positions:
            # Check if the teleportable position is on the line between agent and target
            if self._is_on_line(agent_position, target_position, pos):
                distance = get_distance(agent_position, pos)  # Use get_distance function
                if distance < min_distance:
                    closest_position = pos
                    min_distance = distance
    
        return closest_position
    
    
    def _is_on_line(self, agent_position: dict, target_position: dict, point: dict) -> bool:
        """
        Determine if a point is on the line segment between the agent and the target.
    
        Args:
            agent_position (dict): Current position of the agent.
            target_position (dict): Position of the target object.
            point (dict): Teleportable position to check.
    
        Returns:
            bool: True if the point lies on the line, False otherwise.
        """
        # Extract coordinates
        ax, az = agent_position["x"], agent_position["z"]
        tx, tz = target_position["x"], target_position["z"]
        px, pz = point["x"], point["z"]
    
        # Check collinearity using the cross-product method
        cross_product = abs((px - ax) * (tz - az) - (pz - az) * (tx - ax))
        if cross_product > 1e-5:  # Allow for floating-point tolerance
            return False
    
        # Check if the point is within the bounding box of the line segment
        if min(ax, tx) <= px <= max(ax, tx) and min(az, tz) <= pz <= max(az, tz):
            return True
    
        return False
        
    
    def _get_image(self):
        image = Image.fromarray(self._controller.last_event.frame)
        # self.leolaniClient._add_image()
        return image

    def _step(self, direction: str = "MoveAhead", magnitude: float = None) -> None:
        """
        Robot takes one step in given direction. Options are:
            - MoveAhead
            - MoveBack
            - MoveLeft
            - MoveRight

        Returns None
        """
        self._controller.step(
            action=direction,
            moveMagnitude=magnitude
            ) 

        action_attribute = getattr(Action, direction, None)
        if action_attribute is not None:
            self.leolaniClient._add_action(action_attribute)
        else:
            raise AttributeError(f"'Action' object has no attribute '{direction}'")

        self._metadata.append(self._controller.last_event.metadata)

    def _look(self, direction: str = "LookUp") -> None:
        """
        Robot looks up or down. Options are:
        - LookUp
        - LookDown
    
        Returns None
        """
        self._controller.step(
            action=direction,
            degrees=30
            )

        self.leolaniClient._add_action(Action.direction)
        self._metadata.append(self._controller.last_event.metadata)

    def _rotate(self, direction: str, degrees: float = None) -> None:
        """
        Robot turns in given direction (for optional degrees).
        
        Parameters
        ----------
        direction : str
            Direction to turn in. Can be "RotateLeft" or "RotateRight".
        degrees : float, optional
            Degrees to turn. Default is None.
        
        Returns None
        """
        self._controller.step(
            action=direction,
            degrees=degrees
            )
        
        if direction == "RotateLeft":
            self.leolaniClient._add_action(Action.RotateLeft)
        elif direction == "RotateRight":
            self.leolaniClient._add_action(Action.RotateRight)
        self._metadata.append(self._controller.last_event.metadata)

    def _crouch(self):
        """
        Robot crouches.

        Returns None
        """
        self._controller.step(action="Crouch")

        self.leolaniClient._add_action(Action.Crouch)
        self._metadata.append(self._controller.last_event.metadata)

    def _stand(self):
        """
        Robot stands.

        Returns None
        """
        self._controller.step(action="Stand")

        self.leolaniClient._add_action(Action.Stand)
        self._metadata.append(self._controller.last_event.metadata)

    def _teleport(self, position: dict = None, rotation: dict = None, horizon: float = None, standing: bool = None, to_random: bool = False) -> None:
        """
        Robot teleports to random location.
        
        Parameters
        ----------
        position: dict
            The 'x', 'y', 'z' coordinates.
        rotation: num
            The rotation of the agent's body. If unspecified, the rotation of the agent remains the same.
        horizon: Float
            Look up of down. Negative values (e.g. -30) correspond to agent looking up, and vice versa.
        standing: bool
            True for 

        Returns None
        """

        if to_random:
            rotation = dict(x=0, y=random.randint(0, 360), z=0)
            reachable_positions = self._controller.step(action="GetReachablePositions").metadata["actionReturn"]
            position = random.choice(reachable_positions)
        
        params = {"action": "Teleport", "position": position}
        if rotation is not None:
            params["rotation"] = rotation
        if horizon is not None:
            params["horizon"] = horizon
        if standing is not None:
            params["standing"] = standing
            
        self._controller.step(**params)

        self.leolaniClient._add_action(Action.Teleport)
        self._metadata.append(self._controller.last_event.metadata)
    
    def _find_objects_in_sight(self, object_type: str) -> list:
        """
        Finds objects in sight.

        Parameters
        ----------
        object_type : str
            The type of object to find.

        Returns
        -------
        list
            A list of objects in sight.
        """

        # Get objects in sight
        objects_in_sight = [obj for obj in self._controller.last_event.metadata["objects"] if obj["visibility"] == True]

        # Optionally filter by object type
        if object_type:
            objects_in_sight = [obj for obj in objects_in_sight if obj["objectType"] == object_type]

        return objects_in_sight
    
    def _find_all_rooms(self, number=None):
        """
        Create a list of all rooms (based on `roomType` == "Floor") in current scene. 
        Sorted from nearest to furthest.
        
        """
        rooms = [obj for obj in self._controller.last_event.metadata["objects"] if obj["objectType"] == "Floor"]
        rooms.sort(key=lambda room: room['distance'])
        return rooms

    def _find_nearest_center_of_room(self):
        """
        Create a dictionary with "x", "y", "z" coordinates of nearest center of room(s).
        
        Returns:
        --------
        """
        
        rooms = self._find_all_rooms()
        nearest_room = rooms[0]
        center = nearest_room['axisAlignedBoundingBox']['center']
        return center

    def _done(self) -> None:
        """
        The Done action does nothing to the state of the environment. 
        But, it returns a cleaned up event with respect to the metadata.

        Returns None
        """
        self._controller.step(action="Done")

        self._metadata.append(self._controller.last_event.metadata)


