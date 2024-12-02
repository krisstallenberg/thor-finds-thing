from pydantic import BaseModel
from typing import List, Literal, Optional

# Models for describing views

class ObjectDescription(BaseModel):
    name: Optional[str]
    position: Optional[str]
    size: Optional[str]
    texture: Optional[str]
    color: Optional[str]
    additional_information: Optional[List[str]]
    
class ContextObjectDescription(ObjectDescription):
    position_relative_to_target_object: Optional[str]

class RoomDescription(BaseModel):
    possible_room_types: Optional[List[Literal["Living room", "Kitchen", "Bedroom", "Bathroom"]]]
    size: Optional[str]
    additional_information: Optional[List[str]]

class InitialDescription(BaseModel):
    target_object: Optional[ObjectDescription]
    objects_in_context: Optional[List[ContextObjectDescription]]
    room_description: Optional[RoomDescription]
    additional_information: Optional[List[str]]
    
class ViewDescription(BaseModel):
    objects: Optional[List[ObjectDescription]]
    room_description: Optional[RoomDescription]
    additional_information: Optional[List[str]]

# Models for workflow

class ClarifyingQuestion(BaseModel):
    prompt: str
    relates_to: str 

class ListOfClarifyingQuestions(BaseModel):
    questions: List[ClarifyingQuestion]