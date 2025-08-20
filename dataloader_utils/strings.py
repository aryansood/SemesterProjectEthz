def training_prompt(traj_past, intent, traj_fut):

    # prompt_to_pass = """
    # You are an expert Driver.
    # - You will see a set of past 4 frames each made of the front 5 cameras merged togheter to form a front panoramic view.
    # - You are given the past 4 second trajectory here as array [[x1,y1], [x2, y2], ..., [x16, y16]].
    # Here is the past 4 second trajectory: """+traj_past+"""
    # - You are also given the high level intent from the navigator : """+intent+"""
    # - The objective for you is to undestand the scene you are in and predidct the next 5 second trajectory to avoid hazards you will encounter on the road.
    # - You have to pay attention on nearby confltting vehicle(are there on the any? are there on the front? on the left ? on the rigth? are they going to a be hazard? Do you have to adjust future trajectory either to the right or the left to avoid them?)
    # - You have to pay attention on nearby pedestrian(are there on the any? are there on the front? on the left ? on the rigth? are they going to a be hazard?Do you have to adjust future trajectory either to the right or the left to avoid them?)
    # - You have to pay attention to line roads(The next trajectory is it inside the line?)
    # - You have to pay attention to cyclist(are there on the any? are there on the front? on the left ? on the rigth? are they going to a be hazard?Do you have to adjust future trajectory either to the right or the left to avoid them?)
    # - You have to pay attention to aniamls(are there on the any? are there on the front? on the left ? on the rigth? are they going to a be hazard?Do you have to adjust future trajectory either to the right or the left to avoid them?)
    # - You have to pay attention to door_opening_vehicle(are there on the any? the left ? on the rigth? are they going to a be hazard?Do you have to adjust future trajectory either to the right or the left to avoid them?)
    # - You have to pay attention to weather_condition(are there any problematic weather condition? can you mantain speed?or do you have to decelaaret)
    # - You have to pay attention to construction site and elements(are there any construction elelnts you have to avoid? Do you have to adjust your speed? Do you have to adjust future trajectory either to the right or the left to avoid them?)
    # - Can you accelerate? can the the next rajectory matain same speed? Are there any traffic ligths or sign where you have to stop?.
    # You have to output the future trajectory sampled at 1HZ for the next 5 seconds, Starting at 0.25 so you actually output 6 waypoint.
    # Output Format:
    # {
    # [x0,y0], [x1, y1], [x2, y2], [x3, y3], [x4, y4], [x5, y5]
    # }
    # You have to substitute x0,y0,x1,y1,x2,y2,x3,y3,x4,y4,x5,y5 with appropriate numerical values for obtaining the optimal next 5 seconds trajectory, considering all the elements in the scene.
    # Please the output format should be:
    # {
    # [x0,y0], [x1, y1], [x2, y2], [x3, y3], [x4, y4], [x5, y5]
    # }
    # JUST OUTPUT THE NUMERICAL TRAJECTORY AS TOLD
    # """
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












#def labeler_prompt(traj_past, traj_expert, traj_init, intent):
    # prompt_labeler = """
    # You are an expert labeller of driving scenarios.
    # Input:
    # - 4 frames of multi-view images collected from the ego-vehicle.
    # - Current high-level ("""+intent+""")
    # - 4-second past trajectory """+traj_past+"""
    # - Expert 5-second future trajectory """+traj_expert+""" on the last frames it is drawn in RED, the red trajectory is the Expert one. 
    # - Possible fututre trajectory NOT expert one on the last frames(which of course is the same for the red trajectory one) it is drawn in GREEN, the Green trajectory is the NOT expert one. The Frame with trajectory drawn in GREEN will be passed ad the frame before the red one. 
    # Task:
    # 1. Inspect the input and decide, for each object class below, whether at least one critical
    # instance of that class is present (i.e., it materially affects the ego-vehicle’s future trajectory
    # ). A vehicle can be a car, bus, truck, motorcyclist, scooter, etc. traffic_element includes
    # traffic signs and traffic lights. road_hazard may include hazardous road conditions, road debris,
    # obstacles, etc. A conflicting_vehicle is a vehicle that may potentially conflict with the ego’s
    # future path.
    # Object classes to audit:
    # - nearby_vehicle
    # - pedestrian
    # - cyclist
    # - construction
    # - traffic_element
    # - weather_condition
    # - road_hazard
    # - emergency_vehicle
    # - animal
    # - special_vehicle
    # - conflicting_vehicle
    # - door_opening_vehicle
    # 2. Output "yes" or "no" for every class (no omissions).
    # 3. Compose a concise natural-language description explaining why the expert safe driver plans the
    # given future trajectory.
    # - Mention only the classes you marked "yes"
    # - Describe how each of those critical objects or conditions influences the trajectory.
    # - Do not invent objects or conditions not present in the input.
    # 4. From the expert’s 5-second future trajectory, assign exactly one category from each list:
    # - speed ∈ { keep, accelerate, decelerate }
    # - command ∈ { straight, yield, left_turn, right_turn, lane_follow, lane_change_left,
    # lane_change_right, reverse }
    # Choose the label that best summarises the overall behaviour of the expert future trajectory.
    # - If none fits, use ‘other‘, but do this sparingly.
    # Output format (strict JSON, no extra keys, no commentary):
    # {
    # "critical_objects": {
    # "nearby_vehicle": "yes | no",
    # "pedestrian": "yes | no",
    # "cyclist": "yes | no",
    # "construction": "yes | no",
    # "traffic_element": "yes | no",
    # "weather_condition": "yes | no",
    # "road_hazard": "yes | no",
    # "emergency_vehicle": "yes | no",
    # "animal": "yes | no",
    # "special_vehicle": "yes | no",
    # "conflicting_vehicle": "yes | no",
    # "door_opening_vehicle": "yes | no"
    # },
    # "explanation": "Word description that references the classes marked ’yes’, Also write why NOT expert trajectory is npot as good as the EXPERT ONE(in RED), and how you would adjust the GREEN trajectory to become a better one equal to the OPTIMAL one which is the EXPERT one",
    # Please write detaile desritpion how the car should move how to adjust the trajectory the get close to optimal one(Just do not write only somethingh like we should adjust trajectry to expert one), I nee detailed descritpio on how to adjust.
    # "meta_behaviour": {
    # "speed": "keep | accelerate | decelerate | other",
    # "command": "straight | yield | left_turn | right_turn | lane_follow | lane_change_left |
    # lane_change_right | reverse | other"}}

    # """

    # prompt = """
    # You are an expert labeller of driving scenarios.. Your job is to assess the safety and correctness of the predicted driving trajectory of an autonomous vehicle (shown in GREEN on the final frame), and recommend precise modifications to make it safe and optimal, based on the observed scene and driving intent.

    # You are given:
    # - 4 synchronized camera views from the ego-vehicle (forward, sides, rear), they are given as video last 4 frames.
    # - Current high-level ("""+intent+""")
    # - A 4-second past trajectory of the ego-vehicle."""+traj_past+"""
    # - A 5-second Expert Ground Truth future trajectory"""+traj_expert+""", visualized in RED on the First image you are given in input(The one taken by the Expert Driver).
    # - A A 5-second predicted future trajectory"""+traj_init+""", visualized in GREEN on the IMAGE put in input, given as second image in input after red one( It is an initial guess made by non-expert driver)
    # - A high-level driving intent (e.g., "turn left", "go straight", etc.).

    # Note: The predicted GREEN trajectory is not necessarily wrong — in some cases, it may already be close to optimal and only require minor adjustment for safety, comfort, or compliance. Please assess carefully before suggesting changes.

    # TASKS:

    # 1. Object Presence Assessment  
    # Evaluate each object class below and determine whether any instance materially affects how the ego-vehicle should drive.  
    # Answer with "yes" or "no" for each class:

    #     - nearby_vehicle
    #     - pedestrian
    #     - cyclist
    #     - construction
    #     - traffic_element (signs, lights)
    #     - weather_condition
    #     - road_hazard (e.g., debris, potholes)
    #     - emergency_vehicle
    #     - animal
    #     - special_vehicle (e.g. school bus, utility vehicle)
    #     - conflicting_vehicle (vehicles that cross or block ego path)
    #     - door_opening_vehicle (e.g. parked car with visible opening door)

    # 2. Trajectory Adjustment Reasoning  
    # In natural language:

    # - Explain why the classes marked "yes" are important in this scenario.
    # - Then, describe in detail how the GREEN trajectory must be adjusted to be safe and optimal. 
    # - Be specific about:
    #     - Tell the problem with the Green Trajectory, and potential problem then write how to make green trajectory better.
    #     - Where the green trajectory should curve differently.(The GREEN trajectory can be very wrong, so even major adjustment could be needed)
    #     - When and how speed should change.
    #     - Try to adjust to avoid hazard if there are any.
        

    # Do NOT refer to any expert trajectory or red line.

    # Your correction should match what the optimal trajectory would do, but without referencing it explicitly.
    # Be specific. Only things that will help the trajectory get better. Also keep in mind we the future trajectory is based on current and past information
    # just do not consider possible future behaviour like (" The vehicle should maintain a safe distance from the intersection while stopped and monitor the traffic light status continuously" this is not helping adjusting green trajectory)

    # 3. Meta Behavior Classification  
    # Based on the corrected, optimal trajectory (not the current green one), assign:

    #     "speed": one of { keep | accelerate | decelerate | other }
    #     "command": one of { straight | yield | left_turn | right_turn | lane_follow | lane_change_left | lane_change_right | reverse | other }

    # Choose based on the actual movement needed to complete the intended maneuver safely.

    # OUTPUT FORMAT (strict JSON only):

    # {
    # "critical_objects": {
    #     "nearby_vehicle": "yes | no",
    #     "pedestrian": "yes | no",
    #     "cyclist": "yes | no",
    #     "construction": "yes | no",
    #     "traffic_element": "yes | no",
    #     "weather_condition": "yes | no",
    #     "road_hazard": "yes | no",
    #     "emergency_vehicle": "yes | no",
    #     "animal": "yes | no",
    #     "special_vehicle": "yes | no",
    #     "conflicting_vehicle": "yes | no",
    #     "door_opening_vehicle": "yes | no"
    # },
    # "explanation": "Explain why 'yes' objects are critical. Then, describe in detail how the GREEN trajectory should be modified to create a safe and optimal path. Include specific changes in curvature, timing, speed, alignment, or avoidance behavior based on the scene.",
    # "meta_behaviour": {
    #     "speed": "keep | accelerate | decelerate | other",
    #     "command": "straight | yield | left_turn | right_turn | lane_follow | lane_change_left | lane_change_right | reverse | other"
    # }
    # }
    # """

    # return prompt

