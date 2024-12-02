import chainlit as cl
from llama_index.core.workflow import (
    StartEvent,
    StopEvent,
    Workflow,
    step,
    Event,
    Context
)
import json
import random
from AI2ThorClient import AI2ThorClient
from descriptions import InitialDescription, ViewDescription
from workflow_utils import evaluate_initial_description, generate_questions, populate_initial_description
from openai import OpenAI
from llama_index.llms.openai import OpenAI as OpenAILlamaIndex
from llama_index.llms.ollama import Ollama as OllamaLlamaIndex
from leolani_client import LeolaniChatClient, Action

# Constants
INT_MAX = 2**31 - 1
EMISSOR_PATH = "./emissor"
AGENT = "Human"
HUMAN = "AI2ThorCLient"

class InitialDescriptionComplete(Event):
    pass

class InitialDescriptionIncomplete(Event):
    issues_with_description: list
    structured_description: InitialDescription

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
        chat_mode = cl.user_session.get("chat_profile")
        self.leolaniClient = LeolaniChatClient(emissor_path=EMISSOR_PATH, agent=AGENT, human=HUMAN)
        self.thor = AI2ThorClient(self.leolaniClient, chat_mode)
        
    async def send_message(self, content, author=None, elements=None, actions=None):
        """
        Sends a message using cl.Message. 
        Logs the message in emissor.

        Parameters:
        - content (str): The content of the message (required).
        - author (str, optional): The author of the message. Defaults to None.
        - elements (list, optional): A list of elements to attach to the message. Defaults to None.
        - actions (list, optional): A list of actions to attach to the message. Defaults to None.

        See: https://docs.chainlit.io/api-reference/message

        Returns:
        - The response from cl.Message().send().
        """
        
        message = cl.Message(
            content=content,
            author=author,
            elements=elements,
            actions=actions
        )
        self.leolaniClient._add_utterance(AGENT, content)
        return await message.send() 
   
    async def ask_user(self, content, author=None, elements=None, actions=None):
        """
        Asks the user for input through the chainlit UI.
        
        
        See: https://docs.chainlit.io/api-reference/ask/ask-for-input
        
        Returns:
        - The response from the user, as a Step class.
        (https://docs.chainlit.io/api-reference/step-class)
        """
        
        self.leolaniClient._add_utterance(AGENT, content)
        
        res = await cl.AskUserMessage(content=content, 
                                      author=author,  
                                      timeout=INT_MAX, 
                                      ).send()
        
        if res:
            self.leolaniClient._add_utterance(HUMAN, res['output'])
            return res['output']
    
    @cl.step(type="llm", name="step to evaluate the initial description")
    @step
    async def evaluate_initial_description(self, ev: StartEvent) -> InitialDescriptionComplete | InitialDescriptionIncomplete:
        # Store the initial description in the AI2ThorClient instance and emissor.
        self.thor.initial_description = ev.initial_description
        self.leolaniClient._add_utterance(HUMAN, ev.initial_description)
        
        # Give user a summary of their initial description.
        await self.send_message(content=str(self.thor._llm_ollama.complete(f"Summarize the following description of a view: <description>{ev.initial_description}</description>.\nStart with 'You saw'. Keep your summary short and objective. Don't add any new information, if anything, make it more brief.")))

        # Parse the user input and generate a structured description.
        self.thor.parse_unstructured_description(ev.initial_description)

        # Parse the structured description and generate a list of issues.
        issues = evaluate_initial_description(self.thor.structured_initial_description)

        if not issues:
            return InitialDescriptionComplete()    
        else:
            return InitialDescriptionIncomplete(issues_with_description=issues, structured_description=self.thor.structured_initial_description)

    @cl.step(type="llm", name="step to clarify the initial description")
    @step
    async def clarify_initial_description(self, ev: InitialDescriptionIncomplete) -> InitialDescriptionComplete:
        # List the issues in the initial description to the user.
        await self.send_message(content="Before we move on, I want to ask a few more questions about what you saw. The initial description is incomplete because:\n- " + "\n- ".join([issue for issue, relates_to in ev.issues_with_description]) + "\n\n Once I've gathered this minimal information, let's move on to find the object.")

        # Generate clarifying questions from the list of issues.
        questions = await generate_questions(issues=ev.issues_with_description, 
                                             openai_client=self.thor._llm_openai_multimodal, 
                                             structured_description=json.dumps(ev.structured_description, indent=4, default=str)
                                             )

        # Ask the user clarifying questions.
        clarifying_questions_answers = list()
        for question in questions:
            answer = await self.ask_user(content=question.prompt)
            clarifying_questions_answers.append({
                "question": question.prompt,
                "answer": answer,
                "relates_to": question.relates_to
            })
        
        # For testing purposes, remove later.    
        await self.send_message(content=str(clarifying_questions_answers))
            
        # Populate the initial description using answers to the clarifying questions.
        clarified_structured_description = populate_initial_description(structured_description=ev.structured_description,
                                     openai_client=self.thor._llm_openai_multimodal,
                                     unstructured_description=self.thor.initial_description,
                                     question_answer_pairs=clarifying_questions_answers)

        # Check if the description is complete after clarifying questions were asked.
        issues = evaluate_initial_description(clarified_structured_description)

        if not issues:
            await self.send_message(content=f"Thank you! You answers have given me a better understanding of what to look for.")
            return InitialDescriptionComplete(payload="Description clarified.")    
        else:
            return InitialDescriptionIncomplete(issues_with_description=issues, structured_description=clarified_structured_description)


    @cl.step(type="llm", name="step to find a room of the correct type")
    @step
    async def find_correct_room_type(self, ev: InitialDescriptionComplete | RoomIncorrect | ObjectNotInRoom) -> RoomCorrect | RoomIncorrect:
        current_description = """a living room setup viewed from behind a dark-colored couch. The room has light-colored walls and a floor that seems to be a muted, earthy tone. The main items in the room include:
- A large, dark-colored sofa in the foreground facing a TV.
- A television placed on a small white TV stand, positioned along the far wall.
- A small side table with a blue vase and a decorative item beside the TV stand.
- A wooden shelf or cabinet off to the left side of the room."""
        
        # Generate a structured view description from image.
        self.thor.describe_view_from_image_structured()

        # Generate and stream unstructured view description from image.
        await self.send_message(content=self.thor.describe_view_from_image())
        
        if random.randint(0, 1) == 0:
            return RoomCorrect(payload="Correct room is found.")
        else:
            return RoomIncorrect(payload="Correct room is not found.")

    @cl.step(type="llm", name="step to find the object in the room")
    @step 
    async def find_object_in_room(self, ev: RoomCorrect) -> ObjectInRoom | ObjectNotInRoom:
        if random.randint(0, 10) < 4:
            return ObjectInRoom(payload="Object may be in this room.")
        else:
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
            self.leolaniClient._save_scenario() 
            return StopEvent(result="We found the object!")  # End the workflow
        else:
            if random.randint(0, 1) == 0:
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
        "Hey, there!\n\n We are going to try to find an object together, only through text communication.",
        "To get started, please describe what you saw in detail. Describe the object, its surroundings and what type of room it appeared to be in. Please write in complete sentences."
    ]
    
    for message in intro_messages:
        # Initialize the message with an empty content for streaming
        response_message = cl.Message(content="")
        await response_message.send()
        
        for letter in message:
            response_message.content += letter
            await response_message.update()  
            await asyncio.sleep(0.01) 
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
    await response_message.send()
    
    # Stream the result letter by letter with delays in between
    for letter in result:
        response_message.content += letter
        await response_message.update() 
        await asyncio.sleep(0.01)

@cl.on_chat_end
def end():
    print("Goodbye", cl.user_session.get("id"))
