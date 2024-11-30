from ai2thor.controller import Controller
from PIL import Image
from llama_index.llms.openai import OpenAI as OpenAILlamaIndex
from llama_index.llms.ollama import Ollama as OllamaLlamaIndex
from descriptions import InitialDescription, ViewDescription
from openai import OpenAI
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

    def __init__(self):
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
        self.metadata = []
        self.descriptions = []
        self.unstructured_descriptions = []
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
                {"role": "system", "content": "Your task is to turn a user's description of an object, its context and the room type into a structured response. When information is missing from the user's description, do not make up parts of the description, go ONLY off of the user's description."},
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

    def find_and_go_to_object(self, targets):
        """
        Rotate to locate the target object, then move toward it.
        If the target is not found after 4 rotations, teleport to a random location.
        Args:
            target (str): Name of the target object.
            described (list): List of described objects to assist in finding the target.
        Returns:
            bool: True if the target object is in sight.
            string: Type of object in visible distance.
        """
        turn_angle = None  
        count_of_turns = 0  
        target = targets['target_object']['name']
        context = []
        for item in targets['object_in_context']:
            context.append(item['name'])
        while turn_angle is None:
            image = self._get_image()  # Update the image after each rotation


            # Define target and described objects
            target_objects = {'target': target, 'context': context}

            # Run the workflow to find objects and calculate turn angles
            turn_angle, detections_df, objects_selected = find_objects_and_angles(
                image_path=image,
                target_objects=target_objects
            )
            self.rotate(direction='left')  # Rotate to search for objects
            count_of_turns += 1
            if count_of_turns == 4:  # If the object is not found after 4 rotations
                self._teleport(to_random=True)
                print("Target not found after 4 rotations. Teleporting...")
                return False  # Target not found after teleport

        # Rotate to align with the object
        if turn_angle > 0:
            self._rotate(direction='right', degrees=abs(turn_angle))
        else:
            self._rotate(direction='left', degrees=abs(turn_angle))

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

    def _get_image(self):
        return Image.fromarray(self._controller.last_event.frame)
    
    def _step(self, direction: str = "MoveAhead", magnitude: float = None) -> None:
        """
        Robot takes one step in given direction.
    
        Returns None
        """
        self._controller.step(
            action=direction,
            moveMagnitude=magnitude)
        
        self._metadata.append(self._controller.last_event.metadata)
        
    def _look(self, direction: str = "LookUp") -> None:
        """
        Robot looks up or down.
    
        Returns None
        """
        self._controller.step(
            action=direction,
            degrees=30
            )
        
        self._metadata.append(self._controller.last_event.metadata)
    
    def _rotate(self, direction: str, degrees: float = None) -> None:
        """
        Robot turns in given direction.
        
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
        
        self._metadata.append(self._controller.last_event.metadata)

    def _crouch(self):
        """
        Robot crouches.
    
        Returns None
        """
        self._controller.step(action="Crouch")
        
        self._metadata.append(self._controller.last_event.metadata)
        
    def _stand(self):
        """
        Robot stands.
    
        Returns None
        """
        self._controller.step(action="Stand")
        
        self._metadata.append(self._controller.last_event.metadata)
        
    def _teleport(self, position: dict = None, rotation: dict = None, horizon: float = None, standing: bool = None, to_random: bool = False) -> None:
        """
        Robot teleports to random location.
    
        Returns None
        """
        
        if to_random:
            rotation = {"x": random.randint(0, 360), "y": random.randint(0, 360), "z": random.randint(0, 360)}
            positions = self._controller.step(action="GetReachablePositions").metadata["actionReturn"]
            position = random.choice(positions)
            
        self._controller.step(
            action="Teleport",
            position=position,
            rotation=rotation,
            horizon=horizon,
            standing=standing
        )
        
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
    
    def _done(self) -> None:
        """
        The Done action does nothing to the state of the environment. 
        But, it returns a cleaned up event with respect to the metadata.
    
        Returns None
        """
        self._controller.step(action="Done")
        
        self._metadata.append(self._controller.last_event.metadata)
