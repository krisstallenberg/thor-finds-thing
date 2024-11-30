def find_and_go_to_object(self, targets: dict) -> tuple:
    """
    Locate the target object and move toward it.
    If the target is not found initially, go to the middle of the room and retry.
    If the object is still not found, return False.

    Args:
        targets (dict): A dictionary containing:
            - 'target_object': {'name': str} (Target object name).
            - 'context': [{'name': str}] (Optional described objects in the picture).

    Returns:
        tuple: (bool, list) where:
            - bool: True if the target object is reached, False otherwise.
            - list: A list of log messages generated during the process.
    """
    logs = []  # List to collect log messages
    target_label = targets['target_object']['name']
    context_objects = targets.get('context', [])

    # Step 1: Attempt to find and go to the target object
    if not self._attempt_to_find_and_go_to_target(target_label, context_objects, logs):
        logs.append(f"Target '{target_label}' not found. Going to the middle of the room.")
        
        # Step 2: Go to the middle of the room
        thor._teleport(position=thor._find_nearest_center_of_room())
        logs.append("Teleported to the middle of the room.")

        # Step 3: Retry finding and going to the target object
        if not self._attempt_to_find_and_go_to_target(target_label, context_objects, logs):
            logs.append(f"Target '{target_label}' still not found. Returning False.")
            return False, logs  # Target not found even after retrying

    logs.append(f"Target '{target_label}' successfully reached.")
    return True, logs  # Target successfully reached


def _attempt_to_find_and_go_to_target(self, target_label: str, context_objects: list, logs: list) -> bool:
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
    turn_angle = None  
    count_of_turns = 0  

    while turn_angle is None:
        image = self._get_image()  # Update the image after each rotation

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
            return False  # Target not found

    # Map target label to visible objects
    visible_objects = self._find_objects_in_sight()
    matched_objects = self._map_target_to_visible_objects(target_label, visible_objects)

    if not matched_objects:
        logs.append(f"No semantically matching objects found for target '{target_label}'.")
        return False

    # Select the best object based on proximity to described objects
    selected_object = self._select_best_object(matched_objects, context_objects, visible_objects)
    logs.append(f"Matched target '{target_label}' to object: {selected_object}.")

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
                return True  # Target successfully reached

            # Try stepping toward the target
            if not self._step():
                logs.append("Step failed. Calculating closest teleportable position.")
                break  # Exit to handle teleportation

        # Handle teleportation if stepping fails
        teleport_position = self._calculate_closest_teleportable_position(agent_position, target_position)
        if teleport_position is None:
            logs.append("No suitable teleportable position found.")
            return False

        self._teleport(to=teleport_position)
        teleport_count += 1
        logs.append(f"Teleported to {teleport_position}. Resuming movement.")

    logs.append("Max teleports reached. Could not reach the target.")
    return False


def _select_best_object(self, matched_objects: list, described_objects: list, visible_objects: list) -> dict:
    """
    Select the best object based on proximity to described objects.

    Args:
        matched_objects (list): List of matched objects with IDs, types, and positions.
        described_objects (list): List of described object names.
        visible_objects (list): Full list of visible objects.

    Returns:
        dict: The selected object.
    """
    if not described_objects:
        # If no described objects, prioritize the closest and most aligned object
        matched_objects.sort(key=lambda obj: (abs(obj["relative_position"]["angle"]), obj["relative_position"]["distance"]))
        return matched_objects[0]

    # Find described object positions
    described_positions = [
        obj["position"] for obj in visible_objects if obj["label"] in described_objects
    ]

    if not described_positions:
        print("No described objects found. Defaulting to closest matched object.")
        matched_objects.sort(key=lambda obj: (abs(obj["relative_position"]["angle"]), obj["relative_position"]["distance"]))
        return matched_objects[0]

    # Select the matched object closest to described objects
    for obj in matched_objects:
        obj["proximity_to_described"] = min(
            calculate_distance(obj["position"], desc_pos) for desc_pos in described_positions
        )

    matched_objects.sort(key=lambda obj: obj["proximity_to_described"])
    return matched_objects[0]


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
    # Assuming yaw (rotation around the vertical axis) determines forward direction
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




def _find_objects_in_sight(self, object_type: str) -> list:
    """
    Finds objects in sight and updates the global dictionary with newly detected objects.

    Parameters
    ----------
    object_type : str
        The type of object to find.
    global_dict : dict
        The global dictionary to update with newly detected objects.

    Returns
    -------
    list
        A list of objects in sight.
    """
    # Get objects in sight
    objects_in_sight = [
        obj for obj in self._controller.last_event.metadata["objects"] if obj["visible"] == True
    ]

    # Optionally filter by object type
    if object_type:
        objects_in_sight = [obj for obj in objects_in_sight if obj["objectType"] == object_type]

    # Update the global dictionary
    for obj in objects_in_sight:

        # Use a unique identifier for the object (e.g., object ID or position)
        object_id = obj.get("objectId") or str(obj.get("position"))  #  maybe no position



        # If the object is not already in the global dictionary, add it
        if object_id not in self.objects_seen:
            # Add a Visited attribute with a default value of 0
            obj["Visited"] = 0
            self.objects_seen[object_id] = obj

    return objects_in_sight




def exit_room(self)  -> tuple[bool, list]:
    """
    Navigate out of the current room by finding and moving through a doorway.
    Updates the self.global_dict with all objects found.

    Returns
    -------
    bool
        True if successfully exited the room, otherwise False.
    """


    logs = []  # List to store log messages
    max_teleports = 10  # Prevent infinite retries
    teleport_count = 0  # Counter for teleports

    while teleport_count <= max_teleports:
        # Step 1: Get the visible objects
        visible_objects = self._find_objects_in_sight(object_type=None)

        # Step 2: Find objects listed in the floor's `receptacleObjectIds`
        floor_object = next((obj for obj in visible_objects if obj["type"] == "Floor"), None)
        if not floor_object:
            logs.append("No floor object found in the visible objects.")
            self._teleport(to=self._find_nearest_center_of_room())
            logs.append("Teleported to the middle of the room and retrying.")
            teleport_count += 1
            continue

        # Extract objects on the floor
        floor_position = floor_object["position"]
        objects_on_floor = floor_object.get("receptacleObjectIds", [])
        doorway_objects = []

        for obj_string in objects_on_floor:
            # Split the object string by '|' to extract details
            object_parts = obj_string.split('|')
            if len(object_parts) == 4:
                # Adjust coordinates relative to the floor position
                obj_data = {
                    "type": object_parts[0],
                    "position": {
                        "x": float(object_parts[1]) + floor_position["x"],
                        "y": float(object_parts[2]) + floor_position["y"],
                        "z": float(object_parts[3]) + floor_position["z"]
                    },
                    "Visited": 0  # Add the Visited attribute
                }

                # Add the object to the global dictionary
                object_id = f"{obj_data['type']}|{obj_data['position']['x']}|{obj_data['position']['y']}|{obj_data['position']['z']}"
                if object_id not in self.global_dict:
                    self.global_dict[object_id] = obj_data

                # Collect Doorway objects for navigation
                if obj_data["type"] == "Doorway":
                    doorway_objects.append(obj_data)

        if not doorway_objects:
            logs.append("No Doorway objects found on the floor.")
            self._teleport(to=self._find_nearest_center_of_room())
            logs.append("Teleported to the middle of the room and retrying.")
            teleport_count += 1
            continue

        # Step 3: Find an unvisited doorway
        target_doorway = next((door for door in doorway_objects if not door.get("Visited")), None)
        if not target_doorway:
            logs.append("All doorways are already visited.")
            return False  # No unvisited doorways found

        logs.append(f"Found an unvisited doorway: {target_doorway}")

        # Step 4: Navigate towards the doorway
        agent_position = self.metadata["position"]
        target_position = target_doorway["position"]

        while True:
            distance = get_distance(agent_position, target_position)

            # Continue moving until 0m distance
            if distance <= 0.0:
                logs.append(f"Reached the doorway: {target_doorway}. Stepping through.")
                self._step()  # Step 1 time
                self._step()  # Step 2 time
                # Mark the doorway as visited in the global dictionary
                object_id = f"{target_doorway['type']}|{target_doorway['position']['x']}|{target_doorway['position']['y']}|{target_doorway['position']['z']}"
                self.global_dict[object_id]["Visited"] = 1
                logs.append(f"Doorway '{target_doorway}' marked as visited in global dictionary.")
                return True, logs

            # Try stepping towards the doorway
            if not self._step():
                logs.append("Step failed. Calculating closest teleportable position.")
                break  # Exit loop to handle teleportation

            # Update agent position after successful step
            agent_position = self.metadata["position"]

        # Handle teleportation if stepping fails
        teleport_position = self._calculate_closest_teleportable_position(agent_position, target_position)
        if teleport_position is None:
            logs.append("No suitable teleportable position found.")
            self._teleport(to=self._find_nearest_center_of_room())
            logs.append("Teleported to the middle of the room and retrying.")
            teleport_count += 1
            continue

        # Teleport closer to the doorway and retry
        self._teleport(to=teleport_position)
        teleport_count += 1
        logs.append(f"Teleported to {teleport_position}. Resuming movement towards doorway.")

    # If we exceed max teleports, fail
    logs.append("Max teleports reached. Could not exit the room.")
    return False, logs

