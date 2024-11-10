from ai2thor.controller import Controller
from PIL import Image
from llama_index.llms.openai import OpenAI as OpenAILlamaIndex
from llama_index.llms.ollama import Ollama as OllamaLlamaIndex
from openai import OpenAI
import random
import time
from ThorUtils import encode_image

class AI2ThorClient: 
    """
    An AI2Thor instance with methods wrapping its controller.
    """

    def __init__(self):
        self._controller = Controller(
            agentMode="default",
            visibilityDistance=1.5,
            scene="FloorPlan212",

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
        self._llm_ollama = OllamaLlamaIndex(model="llama3.2", request_timeout=120.0)
        self._llm_openai = OpenAILlamaIndex(model="gpt-4o-2024-08-06")
        self._llm_openai_multimodal = OpenAI()
        

    def describe_scene_from_image(self):
        """
        Describes the current scene using an image-to-text model.
    
        Returns
        -------
        str
            A string describing the current scene.
        """
        
        image = self._get_image()
        
        img_path  = f"log/img/{str(time.time())}.jpg"
        image.save(img_path)
        
        encoded_image = encode_image(img_path)
        
        response = self._llm_openai_multimodal.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Imagine this is your point-of-view. Describe what you see in this virtual environment. Write from the first perspective so start your message with 'I'. First, describe the objects, their colors, and their positions. Don't introduce your description. Start describing directly e.g. 'I currently see a <object> on a <surface> ...'. Be objective in your description! Finally describe the room type: it's either a living room, kitchen, bedroom, or bedroom. It can't be anything else. If you can't infer the room type, just say so. ",
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

        return response.choices[0].message.content        

    def _get_image(self):
        return Image.fromarray(self._controller.last_event.frame)
   
    def _infer_room_type(self):
        """
        Infers the room type from the current scene.
        
        Inference is based on:
        - The image-to-text description of the scene.
        - The objects in the metadata.
        - The AI2Thor object types mapping (https://ai2thor.allenai.org/ithor/documentation/objects/object-types).
        
        Returns
        -------
        Returns a string representing the likely room type.
        """
        pass 
    
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
    
    def _done(self) -> None:
        """
        The Done action does nothing to the state of the environment. 
        But, it returns a cleaned up event with respect to the metadata.
    
        Returns None
        """
        self._controller.step(action="Done")
        
        self._metadata.append(self._controller.last_event.metadata)
