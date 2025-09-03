def training_prompt_waymo(intent, traj_past):
    prompt_to_pass = """
    You are an expert driver.
    Input:
    - 4 frames of multi-view images images collected from the ego-vehicle over the last second
    - Current high-level intent """+intent+"""
    - 4-second past trajectory (16 steps at 4 Hz)"""+traj_past+"""
    Task 1: Critical Objects and Conditions Detection
    Decide whether at least one critical instance of each class could influence the ego-vehicle’s
    future path (no omissions). A vehicle can be a car, bus, truck, motorcyclist, scooter, etc.
    traffic_element includes traffic signs and traffic lights. road_hazard may include hazardous road
    conditions, road debris, obstacles, etc. A conflicting_vehicle is a vehicle that may potentially
    conflict with the ego’s future path. Output "yes" or "no" for every class (no omissions).
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
    } }
    Task 2: Natural Language Explanation
    Compose a concise natural-language description of the optimal future 5-second trajectory for the
    ego vehicle that the expert driver (you) plans and explain why the expert driver plans to execute
    this trajectory.
    - Mention only the classes you marked "yes" in the previous task.
    - Describe how each of those critical objects or conditions influences the optimal trajectory.
    - Do not invent objects or conditions not present in the input.
    Output format (strict JSON, no extra keys, no commentary):
    {
    "explanation": "100-word description that references only the classes marked ’yes’"
    }
    Task 3: Meta-Behaviour Selection
    Assign exactly one category from each list. Choose the label that best summarises the overall
    behaviour of the optimal future trajectory:
    - speed ∈ { keep, accelerate, decelerate }
    - command ∈ { straight, yield, left_turn, right_turn, lane_follow, lane_change_left,
    lane_change_right, reverse }
    - If none fits, use ‘other‘, but do this sparingly.
    Output format (strict JSON, no extra keys, no commentary):
    {
    "meta_behaviour": {
    "speed": "keep | accelerate | decelerate | other",
    "command": "straight | yield | left_turn | right_turn | lane_follow | lane_change_left |
    lane_change_right | reverse | other"
    }}
    Task 4: Future Trajectory Prediction
    Given the input, critical objects/conditions, natural language explanation, and meta-behaviour,
    predict the optimal 5-second future trajectory (6 steps at 1 Hz, the first point rappresent second 0.25) of the ego vehicle.
    Output format (raw text, not markdown or LaTeX):
    Output format Json as follows(raw text, not markdown or LaTeX):
    {
    "traj_fut": [[x_1, y_1], [x_2, y_2], [x_3, y_3], [x_4, y_4], [x_5, y_5], [x_6, y_6]]
    }
    """
    return prompt_to_pass


def training_prompt_covla(intent, traj_past):
    prompt_to_pass = """
    You are an expert left-hand-side driver.
    Input:
    - 4 frames of front-view images images collected from the ego-vehicle over the last second
    - Current high-level intent """+intent+"""
    - 4-second past trajectory (16 steps at 4 Hz)"""+traj_past+"""
    Task 1: Critical Objects and Conditions Detection
    Decide whether at least one critical instance of each class could influence the ego-vehicle’s
    future path (no omissions). A vehicle can be a car, bus, truck, motorcyclist, scooter, etc.
    traffic_element includes traffic signs and traffic lights. road_hazard may include hazardous road
    conditions, road debris, obstacles, etc. A conflicting_vehicle is a vehicle that may potentially
    conflict with the ego’s future path. Output "yes" or "no" for every class (no omissions).
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
    } }
    Task 2: Natural Language Explanation
    Compose a concise natural-language description of the optimal future 5-second trajectory for the
    ego vehicle that the expert driver (you) plans and explain why the expert driver plans to execute
    this trajectory.
    - Mention only the classes you marked "yes" in the previous task.
    - Describe how each of those critical objects or conditions influences the optimal trajectory.
    - Do not invent objects or conditions not present in the input.
    Output format (strict JSON, no extra keys, no commentary):
    {
    "explanation": "100-word description that references only the classes marked ’yes’"
    }
    Task 3: Meta-Behaviour Selection
    Assign exactly one category from each list. Choose the label that best summarises the overall
    behaviour of the optimal future trajectory:
    - speed ∈ { keep, accelerate, decelerate }
    - command ∈ { straight, yield, left_turn, right_turn, lane_follow, lane_change_left,
    lane_change_right, reverse }
    - If none fits, use ‘other‘, but do this sparingly.
    Output format (strict JSON, no extra keys, no commentary):
    {
    "meta_behaviour": {
    "speed": "keep | accelerate | decelerate | other",
    "command": "straight | yield | left_turn | right_turn | lane_follow | lane_change_left |
    lane_change_right | reverse | other"
    }}
    Task 4: Future Trajectory Prediction
    Given the input, critical objects/conditions, natural language explanation, and meta-behaviour,
    predict the optimal 5-second future trajectory (6 steps at 1 Hz, the first point rappresent second 0.25) of the ego vehicle.
    Output format Json as follows(raw text, not markdown or LaTeX):
    {
    "traj_fut": [[x_1, y_1], [x_2, y_2], [x_3, y_3], [x_4, y_4], [x_5, y_5], [x_6, y_6]]
    }
    """
    return prompt_to_pass


def training_prompt_covla_direct_traj(intent, traj_past):
    prompt_to_pass = """
    You are an expert left-hand-side driver.
    Input:
    - 4 frames of front-view images images collected from the ego-vehicle over the last second
    - Current high-level intent """+intent+"""
    - 4-second past trajectory (16 steps at 4 Hz)"""+traj_past+"""
    Task: Future Trajectory Prediction
    Given the input, intent and past trajectory,
    predict the optimal 5-second future trajectory (6 steps at 1 Hz, the first point rappresent second 0.25) of the ego vehicle.
    Output format Json as follows(raw text, not markdown or LaTeX):
    {
    "traj_fut": [[x_1, y_1], [x_2, y_2], [x_3, y_3], [x_4, y_4], [x_5, y_5], [x_6, y_6]]
    }
    """
    return prompt_to_pass

def training_prompt_waymo_direct_traj(intent, traj_past):
    prompt_to_pass = """
    You are an expert driver.
    Input:
    - 4 frames of multi-view images images collected from the ego-vehicle over the last second
    - Current high-level intent """+intent+"""
    - 4-second past trajectory (16 steps at 4 Hz)"""+traj_past+"""
    Task: Future Trajectory Prediction
    Given the input, intent and past trajectory,
    predict the optimal 5-second future trajectory (6 steps at 1 Hz, the first point rappresent second 0.25) of the ego vehicle.
    Output format Json as follows(raw text, not markdown or LaTeX):
    {
    "traj_fut": [[x_1, y_1], [x_2, y_2], [x_3, y_3], [x_4, y_4], [x_5, y_5], [x_6, y_6]]
    }
    """
    return prompt_to_pass