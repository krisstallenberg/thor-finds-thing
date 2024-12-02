from typing import List, Optional
from pydantic import BaseModel, Field
from typing_extensions import Literal
from descriptions import InitialDescription, ListOfClarifyingQuestions

def evaluate_initial_description(structured_description: InitialDescription) -> List[str]:
    """
    Checks if an instance of InitialDescription is sufficiently populated
    and provides specific feedback for missing elements.

    Args:
        structured_description (InitialDescription): The structured description to evaluate.

    Returns:
        List[str]: A list of strings detailing the missing information.
    """
    issues = []

    # Check the room description
    room_desc = structured_description.room_description
    if not room_desc:
        issues.append("The room description is missing entirely.")
    else:
        missing_room_attributes = []
        if not room_desc.room_type:
            missing_room_attributes.append("room type")
        if not room_desc.size:
            missing_room_attributes.append("size")
        if missing_room_attributes:
            issues.append((f"The description of the room is missing {', '.join(missing_room_attributes)}.", "room_description"))

    # Check the context objects
    context_objects = structured_description.objects_in_context or []
    if len(context_objects) < 4:
        issues.append((f"There are only {len(context_objects)} context objects described. We would like a minimum of 4.", "objects_in_context"))
    for i, obj in enumerate(context_objects, start=1):
        missing_context_attributes = []
        if not obj.position_relative_to_target_object:
            missing_context_attributes.append("position relative to the target object")
        if missing_context_attributes:
            object_name = obj.name or f"Context object {i}"
            issues.append((f"The {object_name} is missing {', '.join(missing_context_attributes)}.", f"objects_in_context.{object_name}"))
    
    # Check the target object
    target_object = structured_description.target_object
    if not target_object:
        issues.append(("The target object description is missing entirely.", "target_object"))
    else:
        missing_object_attributes = []
        if not target_object.position:
            missing_object_attributes.append("position")
        if not target_object.size:
            missing_object_attributes.append("size")
        if not target_object.texture:
            missing_object_attributes.append("texture")
        if not target_object.color:
            missing_object_attributes.append("color")
        if missing_object_attributes:
            object_name = target_object.name or "The target object"
            issues.append((f"The target object ({object_name}) is missing {', '.join(missing_object_attributes)}.", "target_object"))

    return issues

async def generate_questions(issues, openai_client, structured_description):
    """
    Takes a list of issues and returns questions to ask the user.
    
    Args:
        issues (Tuple): issues in the initial description. 
        openai_client (OpenAI): OpenAI client for LLM inference.
    """
    
    md_list_of_issues = '\n- '.join([f'{issue} (relates to: `{relates_to}`)' for issue, relates_to in issues])
    
    response = openai_client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": rf"""A user and robotic system are paired up to locate an object in a virtual space. 
                            The human has been shown a target object in a virtual room. 
                            The user has attempted to describe the target object, objects in its context, and the room. 
                            
                            However, there are a number of issues with the description:
                            <List of issues>
                            - {md_list_of_issues}
                            </List of issues>
                            
                            Return a list of clarifying questions the system should ask to solve the issues in the description.
                            
                            Here is the initial user description:
                            <Initial user description>
                            {structured_description}
                            </Initial user description>
                            """,
                        },
                    ],
                },
            ],
            response_format=ListOfClarifyingQuestions,
        )
    
    return response.choices[0].message.parsed.questions


def populate_initial_description(
    structured_description: InitialDescription,
    openai_client,
    unstructured_description: str, 
    question_answer_pairs: List[dict]
) -> InitialDescription:
    """
    Populates missing fields in the structured description by merging with 
    unstructured description and additional inputs from the user.

    Parameters:
        structured_description (InitialDescription): The initial structured description.
        unstructured_description (str): The unstructured description provided by the user.
        question_answer_pairs (List[dict]): List of questions and corresponding user answers.

    Returns:
        InitialDescription: The updated structured description.
    """
    # Use a language model to refine and fill in missing data
    response = openai_client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[
            {
                "role": "system",
                "content": (
                    "Your task is to populate a structured description of a view based on:"
                    " 1. An unstructured user description."
                    " 2. An incomplete structured description (InitialDescription format)."
                    " 3. Additional question-answer pairs to resolve missing information."
                    "\n\nOnly use the information provided. Do not fabricate or assume any details."
                ),
            },
            {"role": "user", "content": f"Unstructured Description: {unstructured_description}"},
            {"role": "user", "content": f"Structured Description: {structured_description.model_dump_json()}"},
            {"role": "user", "content": f"Question-Answer Pairs: {question_answer_pairs}"},
        ],
        response_format=InitialDescription,
    )

    return response.choices[0].message.parsed
    