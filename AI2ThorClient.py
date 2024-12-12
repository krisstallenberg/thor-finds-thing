from ai2thor.controller import Controller
from PIL import Image
from llama_index.llms.openai import OpenAI as OpenAILlamaIndex
from llama_index.llms.ollama import Ollama as OllamaLlamaIndex
from descriptions import InitialDescription, ViewDescription
from leolani_client import Action
from openai import OpenAI
import pandas as pd
import torch
from torchvision.transforms.functional import to_tensor
from PIL import Image, ImageDraw
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np
import math
import random
import base64
import torchvision
from transformers import CLIPProcessor, CLIPModel
import torchvision
from transformers import CLIPProcessor, CLIPModel
import json
import time
from thor_utils import ( 
                        encode_image, 
                        get_distance,
                        map_detected_to_visible_objects,
                        select_objects,
                        expand_box,
                        calculate_turn_angle,
                        compute_final_angle,
                        calculate_turn_and_distance_dot_product
			)

# Constants
VISIBILITY_DISTANCE = 15
SCENE = "FloorPlan209"

class AI2ThorClient: 
    """
    An AI2Thor instance with methods wrapping its controller.
    """

    def __init__(self, leolaniClient, chat_mode: str = "Production", workflow = None):
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
        self.leolaniClient = leolaniClient
        self._llm_ollama = OllamaLlamaIndex(model="llama3.2", request_timeout=120.0)
        self._llm_openai = OpenAILlamaIndex(model="gpt-4o-2024-08-06" )
        self._llm_openai_multimodal = OpenAI( )
        self._chat_mode = chat_mode
        self._workflow = workflow
        self._objects_seen = {}
        self._clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()
        self._frcnn_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True).eval()
        self._clip_processor = clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self._similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
        self._rooms = self._find_all_rooms()
        self._rooms_visited = []

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

    def _get_image(self):
        image = Image.fromarray(self._controller.last_event.frame)
        
        # self.leolaniClient._add_image()
        if self._chat_mode == "Developer":
            self._workflow.send_message(content=image)
        
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
        return self._controller.last_event.metadata['lastActionSuccess']
    
    def _find_objects_in_sight(self, object_type: str = None) -> list:
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
        objects_in_sight = [obj for obj in self._controller.last_event.metadata["objects"] if obj["visible"] == True]

        # Optionally filter by object type
        if object_type:
            objects_in_sight = [obj for obj in objects_in_sight if obj["objectType"] == object_type]

        for obj in objects_in_sight:

            # Use a unique identifier for the object (e.g., object ID or position)
            object_id = obj["objectId"]  #  maybe no position
    
    
    
            # If the object is not already in the global dictionary, add it
            if object_id not in self._objects_seen.keys():
                # Add a Visited attribute with a default value of 0
                obj["visited"] = 0
                self._objects_seen[object_id] = obj

        return objects_in_sight
    
    def _find_all_rooms(self, number=None):
        """
        Create a list of all rooms (based on `roomType` == "Floor") in current scene. 
        Sorted from nearest to furthest.
        
        """
        rooms = [obj for obj in self._controller.last_event.metadata["objects"] if obj["objectType"] == "Floor"]
        rooms.sort(key=lambda room: room['distance'])
        return rooms
    
    def find_nearest_reachable_position(self, destination) -> dict:
        """
        Find a reachable position that is nearest to the given destination.
        
        Parameters
        ----------
        destination: dict
            A dictionary of x, y, z coordinates of the destination.
            
        Returns
        -------
        dict:
            a dictionary of x, y, z coordinates representing the nearest reachable position.
        """
        pass    

    async def _teleport_to_nearest_new_room(self) -> str:
        """
        Teleports the agent to the center of the nearest room if reachable.
        If not, teleports to the nearest reachable position to the center.

        Returns
        -------
        str
            The `objectId` of the room teleported to.
        """      
        # Find all rooms
        rooms = self._find_all_rooms()
        
        # Initialize destination_room with None
        destination_room = None
        
        # Handle single room scenes
        if len(rooms) == 1 and rooms[0]['objectId'] in self._rooms_visited:
            await self._workflow.send_message(content=f"I've looked in all rooms now...")
            return False
        
        # Communicate what's happening to the user
        if self._rooms_visited == []:
            await self._workflow.send_message(content=f"I'm going to the center of the current room.")
        else:
            await self._workflow.send_message(content=f"I'm going to look for a room I haven't visited yet.")
        
        # Iterate over rooms to find nearest non-visited room
        for room in rooms:
            if room['objectId'] not in self._rooms_visited:
                destination_room = room
                break

        # If no room is found, all rooms have been visited
        if destination_room is None:
            await self._workflow.send_message(content=f"I've looked in all rooms now...")
            return False
        elif self._rooms_visited != []:
            await self._workflow.send_message(content=f"I found a new room. I'm going there right now.")

        # Append the nearest non-visited room to rooms_visited
        self._rooms_visited.append(destination_room['objectId'])
         
        # Find the nearest non-visited room's center
        center = destination_room['axisAlignedBoundingBox']['center']

        # Get globally reachable positions
        reachable_positions = self._controller.step(action="GetReachablePositions").metadata["actionReturn"]
        
        # Teleport as close to the center of the nearest non-visited room as possible
        closest_reachable_position = find_closest_position(reachable_positions, center)
        return self._teleport(position=closest_reachable_position)

    def _done(self) -> None:
        """
        The Done action does nothing to the state of the environment. 
        But, it returns a cleaned up event with respect to the metadata.

        Returns None
        """
        self._controller.step(action="Done")

        self._metadata.append(self._controller.last_event.metadata)


    def _find_objects_and_angles(self, image, target_objects):
        """Main workflow to find objects and compute turn angles."""

        # Detection
        detections, image_width, image_height = self._detect_objects(image)

        # Classification
        detections_df = self._classify_objects(detections, image, target_objects)

        # detections_df = pd.DataFrame(detections_with_labels)
        if len(detections_df) == 0:
            return None, None, None

        # Object Selection
        object_role, selected_objects = select_objects(detections_df, target_objects)

        # Angle Calculation
        turn_angle = compute_final_angle(selected_objects, image_width)

        return turn_angle, detections_df, selected_objects



    def _attempt_to_find_and_go_to_target(self, target_label: str, context_objects: list, logs: list, agent_into :tuple) -> tuple:
        """
        Attempt to locate the target object and move toward it.
        If the agent cannot step, teleport closer to the target.

        Args:
            target_label (str): Name of the target object.
            context_objects (list): List of described object names.
            logs (list): A list to collect log messages.

        Returns:
            bool: True if the target object is reached, False otherwise.
        """
        turn_angle=None
        matched_object=None
        count_of_turns = agent_info[0]
        agent_rot=agent_info[2]
        agent_pos=agent_info[1]

        if agent_rot != None or agent_pos != None:

            self._teleport(position=agent_pos, rotation=agent_rot, horizon=0)
            

        # Initialize agent position and rotation
        agent_position = self._metadata[-1]['agent']['position']
        agent_rotation = self._metadata[-1]['agent']['rotation']
        logs.append(f'Started looking for the {target_label} in the new room!')
        while (turn_angle is None or matched_object is None):
            image = self._get_image()  # Update the image after each rotation
            
            # Define target and described objects
            target_objects = {'target': target_label, 'context': context_objects}
            # Find the object using metadata
            meta_best_match, meta_turn_angle = self._find_target_angle_and_id_with_meta(target_objects['target'])

            # Run the workflow to find objects and calculate turn angles
            turn_angle, detections_df, objects_selected = self._find_objects_and_angles(
                image=image,
                target_objects=target_objects
            )
            visible_objects = self._find_objects_in_sight(object_type=None)

            matched_object, logs = self._select_objects_by_similarity(logs, target_label, visible_objects)
            if matched_object!= None and turn_angle != None:
                logs.append(f'Found a possible object! Trying now to approach it.')

            
            if turn_angle is None and meta_turn_angle is not None:
                turn_angle = meta_turn_angle
                matched_object = meta_best_match
                logs.append(f'Did not find the object through camera, but found one through the Metadata.')

            
            for item in self._objects_seen.values():  # Iterate over the dictionary values

                if matched_object is not None and item['objectId'] == matched_object['id']:
                    if item['visited'] == 1:
                        turn_angle = None
                        matched_object = None
                        print('already seen the object')
                        logs.append(f'The object that was found has already been found before. Continueing the search!')

            if turn_angle is None and meta_turn_angle is None and matched_object != None:
                for item in self._objects_seen.values():
                    if item['objectId'] == matched_object['id']:
                        item['visited'] = 1
                logs.append('The navigation to the found object could not be calculated, but the object ID is known!')
                return matched_object['id'], logs, (count_of_turns, agent_position, agent_rotation)
                        
            if not turn_angle or not matched_object:
                self._rotate(direction='RotateLeft')  # Rotate to search for objects
                agent_position = self._metadata[-1]['agent']['position']
                agent_rotation = self._metadata[-1]['agent']['rotation']
                logs.append(f"Rotated left to search for '{target_label}'.")
                count_of_turns += 1

            if count_of_turns >= 3:
                logs.append(f"Target '{target_label}' not found after 3 rotations.")
                return False, logs, (count_of_turns, agent_position, agent_rotation)
    


        logs.append(f"Matched target '{target_label}' to object: {matched_object}.")
        selected_object = matched_object

        # Rotate to align with the selected object
        if turn_angle > 0:
            self._rotate(direction='RotateRight', degrees=abs(turn_angle))
            logs.append(f"Rotated right by {abs(turn_angle)} degrees to align with the target.")
        else:
            self._rotate(direction='RotateLeft', degrees=abs(turn_angle))
            logs.append(f"Rotated left by {abs(turn_angle)} degrees to align with the target.")

        # Step toward the target object
        max_teleports = 10  # Prevent infinite retries
        teleport_count = 0
        step_count = 0
        max_steps = 20

        while teleport_count <= max_teleports:
            while step_count < max_steps:
                agent_position = self._metadata[-1]['agent']['position']
                target_position = selected_object["position"]

                distance = get_distance(agent_position, target_position)
                print(f'{distance}, distance')

                if distance <= 1.5:
                    logs.append(f"Target '{target_label}' is within {distance:.2f} meters. Successfully reached.")
                    for item in self._objects_seen.values():
                        if item['objectId'] == selected_object['id']:
                            item['visited'] = 1
                    return selected_object['id'], logs, (count_of_turns, agent_position, agent_rotation)

                self._step('MoveAhead')
                logs.append('Stepped forward.')
                step_count += 1

                if not self._metadata[-1]['lastActionSuccess']:
                    logs.append("Step failed. Calculating closest teleportable position.")
                    break

            teleport_position = self._calculate_closest_teleportable_position(agent_position, target_position)
            if teleport_position is None:
                logs.append("No suitable teleportable position found.")
                for item in self._objects_seen.values():
                    if item['objectId'] == selected_object['id']:
                        item['visited'] = 1
                return selected_object['id'], logs, (count_of_turns, agent_position, agent_rotation)

            self._teleport(position=teleport_position)
            teleport_count += 1
            logs.append(f"Teleported to {teleport_position}. Resuming movement.")

        logs.append("Max teleports reached. Could not reach the target.")
        for item in self._objects_seen.values():
            if item['objectId'] == selected_object['id']:
                item['visited'] = 1
        return selected_object['id'], logs, (count_of_turns, agent_position, agent_rotation)



    def _select_objects_by_similarity(self, logs, target_label: str, visible_objects: list, similarity_threshold=0.40) -> list:
        """
        Select the best matching visible object for the target label based on semantic similarity using embeddings.
    
        Args:
            logs (list): Log messages.
            target_label (str): The target label to match.
            visible_objects (list): List of visible objects, each as a dictionary with 'objectType' and other properties.
            similarity_threshold (float): The minimum cosine similarity required to consider a match.
    
        Returns:
            tuple: (Best match object or None, Updated logs)
        """
        try:
            # Step 1: Extract object types and metadata from visible objects
            object_data = [
                {"id": obj["objectId"], "type": obj["objectType"], "position": obj["position"]}
                for obj in visible_objects
            ]
            object_types = [obj["type"] for obj in object_data]
    
            if not object_types:
                logs.append("No object types found in visible objects.")
                return None, logs
    
            # Step 2: Encode target label and object types
            target_embedding = self._similarity_model.encode([target_label])
            object_embeddings = self._similarity_model.encode(object_types)
    
            if target_embedding.size == 0 or len(object_embeddings) == 0:
                logs.append("Embeddings for target or objects are missing or invalid.")
                return None, logs
    
            # Step 3: Compute cosine similarity between target label and object types
            similarities = cosine_similarity(target_embedding, object_embeddings).flatten()
    
            # Step 4: Find the best match based on similarity
            best_match_index = np.argmax(similarities)
            best_similarity = similarities[best_match_index]
    
            if best_similarity < similarity_threshold:
                logs.append(f"No object matches the target label '{target_label}' with sufficient similarity (Threshold: {similarity_threshold}).")
                return None, logs
    
            # Step 5: Return the best match object metadata
            best_match_object = object_data[best_match_index]
            logs.append(f"Best match for '{target_label}' is '{best_match_object['type']}' with similarity {best_similarity:.2f}.")
            return best_match_object, logs
    
        except Exception as e:
            logs.append(f"Error during object similarity computation: {e}")
            return None, logs

    
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
            agent_position (dict): The agent's current position.
            target_position (dict): The target's position.
    
        Returns:
            dict or None: The closest teleportable position if found, otherwise None.
        """
        # Extract teleportable positions from metadata
        teleportable_positions = self._controller.step(action="GetReachablePositions").metadata["actionReturn"]
    
        if not teleportable_positions:
            raise ValueError("No teleportable positions found in the event metadata.")
    
        closest_position = None
        min_distance = float('inf')
    
        for position in teleportable_positions:
            # Calculate Euclidean distance between the agent and each teleportable position
            distance = math.sqrt(
                (position["x"] - target_position["x"]) ** 2 +
                (position["z"] - target_position["z"]) ** 2
            )
            if distance < min_distance:
                min_distance = distance
                closest_position = position
    
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


    
    def _detect_objects(self, image, confidence_threshold=0.60):
        """Detect objects using Faster R-CNN."""
        image_width, image_height = image.size
        tensor_image = to_tensor(image).unsqueeze(0)  # Correct usage

        # Run Faster R-CNN
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




    def _classify_objects(self, detections, image, target_objects, padding=5):
        """Classify detected objects using CLIP."""
        image_width, image_height = image.size
        detections_with_labels = []
    
        # Validate target_objects
        if not isinstance(target_objects, dict) or "target" not in target_objects or "context" not in target_objects:
            raise ValueError("The 'target_objects' dictionary must contain 'target' and 'context' keys.")
    
        for det in detections:
            try:
                # Validate detection keys
                if not all(key in det for key in ["x1", "y1", "x2", "y2"]):
                    raise KeyError(f"Detection is missing bounding box keys: {det}")
                
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
                    text=[target_objects["target"]] + target_objects["context"],
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
                labels = [target_objects["target"]] + target_objects["context"]
                max_prob_idx = torch.argmax(probs).item()
                detected_label = labels[max_prob_idx]
                confidence = probs[max_prob_idx].item()
    
                # Append result if label matches target_label
                if detected_label == target_objects["target"]:
                    object_center_x = (x1 + x2) / 2
                    detections_with_labels.append({
                        "label": detected_label,
                        "confidence": confidence,
                        "center_x": object_center_x,
                        "box": (x1, y1, x2, y2)
                    })
            except Exception as e:
                print(f"Error during object classification: {e}")
                continue
 
        try:
            detections_df = pd.DataFrame(detections_with_labels)
            if detections_df.empty or "label" not in detections_df.columns:
                return pd.DataFrame()  # Return an empty DataFrame
        except Exception as e:
            print(f"Error creating DataFrame: {e}")
            return pd.DataFrame()  # Return an empty DataFrame
    
        return detections_df


    
    def _find_target_angle_and_id_with_meta(self, target_name):
        """
        Finds the best-matching object based on semantic similarity and calculates the turn angle.
    
        Args:
            target_name (str): Name of the target object.
    
        Returns:
            tuple: (best_matching_object_id, turn_angle)
        """
        # Step 1: Get visible objects
        visible_objects = self._find_objects_in_sight(object_type=None)  

        # Step 2: Find the floor object and extract receptacle object IDs
        floor_object = next((obj for obj in visible_objects if obj['objectType'] == 'Floor'), None)
        if not floor_object or 'receptacleObjectIds' not in floor_object:
            return None, None
    
        receptacle_ids = floor_object['receptacleObjectIds']
    
        # Step 3: Filter visible objects to those whose IDs match receptacle IDs
        filtered_objects = [obj for obj in visible_objects if obj['objectId'] in receptacle_ids]
        logs=[]


        best_match, logs = self._select_objects_by_similarity( logs, target_name, filtered_objects)

    
        if not best_match:
            return None, None
    
        # Extract position of the best match
        object_position = best_match['position']  
    
        # Step 5: Get agent's position and rotation 
        agent_position = self._metadata[-1]['agent']['position'] 
        agent_rotation = self._metadata[-1]['agent']['rotation']  
        print(agent_position)
        print(agent_rotation)
        # Step 6: Calculate turn angle and distance 
        turn_angle, _ = calculate_turn_and_distance_dot_product(
            (agent_position['x'], agent_position['z']),  # Use x and z for 2D position
            agent_rotation,
            (object_position['x'], object_position['z'])
        )
    
        # Transform the turn angle into a single float value 
        turn_angle_float = turn_angle
    
        # Step 7: Return the object ID and turn angle
        return best_match, turn_angle_float
    
