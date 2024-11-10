import chainlit as cl
from llama_index.core.workflow import (
    StartEvent,
    StopEvent,
    Workflow,
    step,
    Event,
    Context
)
import random
from AI2ThorClient import AI2ThorClient
from openai import OpenAI
from llama_index.llms.openai import OpenAI as OpenAILlamaIndex
from llama_index.llms.ollama import Ollama as OllamaLlamaIndex

INT_MAX = 2**31 - 1

class InitialDescriptionComplete(Event):
    payload: str

class InitialDescriptionIncomplete(Event):
    payload: str

class ObjectFound(Event):
    payload: str

class WrongObjectSuggested(Event):
    payload: str

class RoomCorrect(Event):
    payload: str

class RoomIncorrect(Event):
    payload: str

class ObjectInRoom(Event):
    payload: str

class ObjectNotInRoom(Event):
    payload: str

class ThorFindsObject(Workflow):
    
    def __init__(self, timeout: int = 10, verbose: bool = False):
        super().__init__(timeout=timeout, verbose=verbose)
        self.thor = AI2ThorClient()
    
    @cl.step(type="llm", name="step to evaluate the initial description")
    @step
    async def evaluate_initial_description(self, ev: StartEvent) -> InitialDescriptionComplete | InitialDescriptionIncomplete:
        await cl.Message(content=str(self.thor._llm_ollama.complete(f"Summarize the following description of a scene: <description>{ev.initial_description}</description>.\nStart with 'You saw'. Keep your summary short and objective. Don't add any new information, if anything, make it more brief."))).send()
        await cl.Message(content=self.thor.describe_scene_from_image()).send()
        
        if random.randint(0, 1) == 0:
            print("Initial description is complete.")
            return InitialDescriptionComplete(payload="Initial description is complete.")
        else:
            print("Initial description is incomplete.")
            return InitialDescriptionIncomplete(payload="Initial description is incomplete.")

    @cl.step(type="llm", name="step to clarify the initial description")
    @step
    async def clarify_initial_description(self, ev: InitialDescriptionIncomplete) -> InitialDescriptionComplete:
        print("Initial description clarified.")
        return InitialDescriptionComplete(payload="Description clarified.")

    @cl.step(type="llm", name="step to find a room of the correct type")
    @step
    async def random_teleport(self, ev: InitialDescriptionComplete | RoomIncorrect | ObjectNotInRoom) -> RoomCorrect | RoomIncorrect:
        current_description = """a living room setup viewed from behind a dark-colored couch. The room has light-colored walls and a floor that seems to be a muted, earthy tone. The main items in the room include:
- A large, dark-colored sofa in the foreground facing a TV.
- A television placed on a small white TV stand, positioned along the far wall.
- A small side table with a blue vase and a decorative item beside the TV stand.
- A wooden shelf or cabinet off to the left side of the room."""
        
        await cl.Message(content="""Before we move on, let me tell you what I see. \n\nI see {}""".format(current_description)).send()
        
        if random.randint(0, 1) == 0:
            print("Teleported to correct room type.")
            return RoomCorrect(payload="Correct room is found.")
        else:
            print("Teleported to incorrect room type.")
            return RoomIncorrect(payload="Correct room is not found.")

    @cl.step(type="llm", name="step to find the object in the room")
    @step 
    async def find_object_in_room(self, ev: RoomCorrect) -> ObjectInRoom | ObjectNotInRoom:
        if random.randint(0, 10) < 4:
            print("Object may be in this room.")
            return ObjectInRoom(payload="Object may be in this room.")
        else:
            print("Object is not in this room.")
            return ObjectNotInRoom(payload="Object is not in this room.")
    
    @cl.step(type="llm" , name="step to suggest an object")
    @step 
    async def suggest_object(self, ev: ObjectInRoom | WrongObjectSuggested) -> WrongObjectSuggested | ObjectNotInRoom | StopEvent:
        
        actions = [
        cl.Action(name="Yes", value="example_value", description="The identifier matches the one of the target object."),
        ]
        
        object_found = await cl.AskActionMessage( 
            content="Does the target object have identifier {} ?".format(random.randint(1000, 9999)),
            actions=[
                cl.Action(name="Yes", value="yes", label="✅ Yes"),
                cl.Action(name="No", value="no", label="❌ No"),
            ],
            timeout=INT_MAX
        ).send()
        
        if object_found.get("value") == "yes":
            return StopEvent(result="We found the object!")  # End the workflow
        else:
            if random.randint(0, 1) == 0:
                print("Wrong object suggested.")
                return WrongObjectSuggested(payload="Couldn't find object in this room.")
            else:
                return ObjectNotInRoom(payload="Object is not in this room.")

import asyncio

@cl.on_chat_start
async def on_chat_start():
    """
    The entry point of the application.
    
    Starts the ChainLit UI and initializes the LlamaIndex workflow.
    
    Returns None
    """
    app = ThorFindsObject(
        verbose=True, 
        timeout=6000
    )  # The app times out if it runs for 6000s without any result
    cl.user_session.set("app", app)
    
    # Introductory messages to be streamed
    intro_messages = [
        "Hey, we are going to try to find an object together, only through text communication. You will be asked to describe an object in a scene, and I will try to find the object.",
        "In order to do this, I will need you to describe the scene in detail. Please write in complete sentences, from the first person perspective.\n\nLet's get started...",
        "Please tell me what you saw."
    ]
    
    for message in intro_messages:
        # Initialize the message with an empty content for streaming
        response_message = cl.Message(content="")
        await response_message.send()
        
        for letter in message:
            response_message.content += letter
            await response_message.update()  
            await asyncio.sleep(0.025) 
        await asyncio.sleep(1) 

@cl.on_message
async def on_message(message: cl.Message):
    """
    The ChainLit message handler that
    - Starts the LlamaIndex workflow
    - Streams the result letter by letter.
    
    Returns None
    """
    app = cl.user_session.get("app")
    result = await app.run(initial_description=message.content)
    
    # Initialize the response message with an empty content
    response_message = cl.Message(content="")
    await response_message.send()  # Send an initial empty message
    
    # Stream the result letter by letter
    for letter in result:
        response_message.content += letter
        await response_message.update()  # Update the message with new content
        await asyncio.sleep(0.025)  # Adjust delay as needed for streaming effect

@cl.on_chat_end
def end():
    print("Goodbye", cl.user_session.get("id"))
