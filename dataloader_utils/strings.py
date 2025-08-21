def training_prompt(traj_past, intent, traj_fut):
    prompt_to_pass = """
    You are an expert labeller of driving scenarios.
    Input:
    - 4 frames of multi-view images collected from the ego-vehicle over the last second
    - Current high-level intent """+intent+"""
    - 4-second past trajectory (16 steps at 4 Hz)"""+traj_past+"""
    - Expert 5-second future trajectory (20 steps at 4 Hz)"""+traj_fut+"""
    Task:
    1. Inspect the input and decide, for each object class below, whether at least one critical
    instance of that class is present (i.e., it materially affects the ego-vehicle’s future trajectory
    ). A vehicle can be a car, bus, truck, motorcyclist, scooter, etc. traffic_element includes
    traffic signs and traffic lights. road_hazard may include hazardous road conditions, road debris,
    obstacles, etc. A conflicting_vehicle is a vehicle that may potentially conflict with the ego’s
    future path.
    Object classes to audit:
    - nearby_vehicle
    - pedestrian
    - cyclist
    - construction
    - traffic_element
    - weather_condition
    - road_hazard
    - emergency_vehicle
    - animal
    - special_vehicle
    - conflicting_vehicle
    - door_opening_vehicle
    2. Output "yes" or "no" for every class (no omissions).
    3. Compose a concise natural-language description explaining why the expert safe driver plans the
    given future trajectory.
    - Mention only the classes you marked "yes"
    - Describe how each of those critical objects or conditions influences the trajectory.
    - Do not invent objects or conditions not present in the input.
    4. From the expert’s 5-second future trajectory, assign exactly one category from each list:
    - speed ∈ { keep, accelerate, decelerate }
    - command ∈ { straight, yield, left_turn, right_turn, lane_follow, lane_change_left,
    lane_change_right, reverse }
    Choose the label that best summarises the overall behaviour of the expert future trajectory.
    - If none fits, use ‘other‘, but do this sparingly.
    Output format (strict JSON, no extra keys, no commentary):
    {
    "critical_objects": {
    "nearby_vehicle": "yes | no",
    "pedestrian": "yes | no",
    "cyclist": "yes | no",
    "construction": "yes | no",
    "traffic_element": "yes | no",
    "weather_condition": "yes | no",
    "road_hazard": "yes | no",
    "emergency_vehicle": "yes | no",
    "animal": "yes | no",
    "special_vehicle": "yes | no",
    "conflicting_vehicle": "yes | no",
    "door_opening_vehicle": "yes | no"
    },
    "explanation": "100-word description that references only the classes marked ’yes’",
    "meta_behaviour": {
    "speed": "keep | accelerate | decelerate | other",
    "command": "straight | yield | left_turn | right_turn | lane_follow | lane_change_left |
    lane_change_right | reverse | other"}}
    """
    return prompt_to_pass


def training_prompt_covla(traj_past, intent, traj_fut):
    prompt_to_pass = """
    You are an expert labeller of driving scenarios.
    Input:
    - 4 frames of multi-view images collected from the ego-vehicle over the last second
    - Current high-level intent """+intent+"""
    - 4-second past trajectory (16 steps at 4 Hz)"""+traj_past+"""
    - Expert 5-second future trajectory (20 steps at 4 Hz)"""+traj_fut+"""
    Task:
    1. Inspect the input and decide, for each object class below, whether at least one critical
    instance of that class is present (i.e., it materially affects the ego-vehicle’s future trajectory
    ). A vehicle can be a car, bus, truck, motorcyclist, scooter, etc. traffic_element includes
    traffic signs and traffic lights. road_hazard may include hazardous road conditions, road debris,
    obstacles, etc. A conflicting_vehicle is a vehicle that may potentially conflict with the ego’s
    future path.
    Object classes to audit:
    - nearby_vehicle
    - pedestrian
    - cyclist
    - construction
    - traffic_element
    - weather_condition
    - road_hazard
    - emergency_vehicle
    - animal
    - special_vehicle
    - conflicting_vehicle
    - door_opening_vehicle
    2. Output "yes" or "no" for every class (no omissions).
    3. Compose a concise natural-language description explaining why the expert safe driver plans the
    given future trajectory.
    - Mention only the classes you marked "yes"
    - Describe how each of those critical objects or conditions influences the trajectory.
    - Do not invent objects or conditions not present in the input.
    4. From the expert’s 5-second future trajectory, assign exactly one category from each list:
    - speed ∈ { keep, accelerate, decelerate }
    - command ∈ { straight, yield, left_turn, right_turn, lane_follow, lane_change_left,
    lane_change_right, reverse }
    Choose the label that best summarises the overall behaviour of the expert future trajectory.
    - If none fits, use ‘other‘, but do this sparingly.
    Output format (strict JSON, no extra keys, no commentary):
    {
    "critical_objects": {
    "nearby_vehicle": "yes | no",
    "pedestrian": "yes | no",
    "cyclist": "yes | no",
    "construction": "yes | no",
    "traffic_element": "yes | no",
    "weather_condition": "yes | no",
    "road_hazard": "yes | no",
    "emergency_vehicle": "yes | no",
    "animal": "yes | no",
    "special_vehicle": "yes | no",
    "conflicting_vehicle": "yes | no",
    "door_opening_vehicle": "yes | no"
    },
    "explanation": "100-word description that references only the classes marked ’yes’",
    "meta_behaviour": {
    "speed": "keep | accelerate | decelerate | other",
    "command": "straight | yield | left_turn | right_turn | lane_follow | lane_change_left |
    lane_change_right | reverse | other"}}
    """
    return prompt_to_pass













