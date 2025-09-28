"""
GSM8K-V Evaluation Prompts for Different Evaluation Modes

This module provides specialized prompt generation functions for various evaluation modes in GSM8K-V:

- Text-only mode: get_text_only_math_prompt() - Generates prompts for pure text mathematical word problems, requiring step-by-step reasoning and final answer extraction.

- Image-based mode: get_image_math_prompt() - Creates prompts for visual math problems with two sub-modes:
  * "implicit" mode: Model analyzes images containing embedded math problems without explicit questions
  * "explicit" mode: Model answers specific questions about provided images with mathematical content

- Scene-based mode: get_scene_math_prompt() - Generates prompts for complex multi-scene mathematical narratives, requiring analysis of interconnected visual scenes with detailed examples and step-by-step reasoning across scenes.
"""

from typing import Dict, List, Any, Optional

def get_text_only_math_prompt(question: str) -> str:
    """
    Generate prompt for text-only math evaluation.
    
    Args:
        question: The math question
        
    Returns:
        Formatted prompt
    """
    format_instruction = """
When providing your final answer:
- If the answer can be expressed as a whole number (integer), provide it as an integer
- If the answer requires decimals and can be expressed with a finite decimal, use decimal format
- Otherwise, use fraction format if appropriate

"""
    
  
    return f"""You are an expert at solving mathematical word problems.
Please solve the following problem step by step, showing your reasoning.

{format_instruction}Problem: {question}

Please think step by step. After your reasoning, output your final answer on a new line starting with "FINAL ANSWER: " followed by the number only.
"""
 
 
def get_image_math_prompt(question: str, prompt_mode: str = "implicit") -> str:
    """
    Generate prompt for image-based math evaluation.
    
    Args:
        question: The specific math question or modify_scene_related_question
        prompt_mode: Either "implicit" or "explicit"
        
    Returns:
        Formatted prompt
    """
    format_instruction = """
When providing your final answer:
- If the answer can be expressed as a whole number (integer), provide it as an integer
- If the answer requires decimals and can be expressed with a finite decimal, use decimal format
- Otherwise, use fraction format if appropriate

"""
    
    if prompt_mode == "implicit":
        return f"""You are an expert at solving mathematical problems based on visual information.
I'll show you some images that contain a math problem or story.

{format_instruction}Answer the question in the images.

Please think step by step. After your reasoning, output your final answer on a new line starting with "FINAL ANSWER: " followed by the number only.
"""

    elif prompt_mode == "explicit":
        return f"""You are an expert at solving mathematical problems based on visual information.
I'll show you one or more images related to a math problem, along with a question.
Please analyze the images and answer the math question step by step.

{format_instruction}Question to answer: {question}

Please think step by step. After your reasoning, output your final answer on a new line starting with "FINAL ANSWER: " followed by the number only.
"""


def get_scene_math_prompt(scene_description: str) -> str:
    """
    Generate prompt for scene-based math evaluation.

    Args:
        scene_description: The complete scene description with multiple scenes and final question

    Returns:
        Formatted prompt for scene understanding and math problem solving
    """

    format_instruction = """
When providing your final answer:
- If the answer can be expressed as a whole number (integer), provide it as an integer
- If the answer requires decimals and can be expressed with a finite decimal, use decimal format
- Otherwise, use fraction format if appropriate

"""

    example = """
Example1:
Input: "Scene 1: Objects: - 9 eggs, each with a smooth, off-white shell and slightly varied size, clustered together in a shallow, straw-lined duck nest.\n- Janet (white woman, shoulder-length brown hair, wearing a light blue blouse and khaki pants) standing beside the nest.\n- A speech bubble above Janet’s head stating: 'besides these, there are 7 more eggs in the nest.' Composition: - The duck nest is placed in the lower center of the scene, with the 9 eggs arranged in a loose, non-overlapping cluster inside the straw nest, each egg clearly visible and distinct.\n- Janet stands to the right of the nest, facing slightly toward it, her body angled to show engagement with the eggs.\n- The speech bubble is positioned above Janet’s head, with its tail pointing directly to her mouth, and does not overlap with Janet or the nest.\n- All objects mentioned in the scene are clearly visible and not overlapping; each is distinctly separated so that every object can be easily identified. Action: Janet extends her right arm and points directly at the cluster of eggs in the nest, visually anchoring her statement to the eggs. Scene 2: Objects: - 3 eggs, each with a smooth, off-white shell, arranged on a white ceramic breakfast plate.\n- Janet (white woman, same appearance as before) seated at a wooden dining table.\n- A speech bubble above Janet’s head stating: 'I love eggs for breakfast.' Composition: - The breakfast plate is placed in the center of the table, with the 3 eggs spaced evenly on the plate, none overlapping.\n- Janet is seated on the left side of the table, facing the plate, holding a fork in her right hand and a gentle smile on her face.\n- The speech bubble is above Janet’s head, with its tail pointing to her mouth, and does not overlap with Janet or the plate.\n- All objects mentioned in the scene are clearly visible and not overlapping; each is distinctly separated so that every object can be easily identified. Action: Janet sits at the table, holding her fork, and looks at the eggs on her plate while eating. Scene 3: Objects: - 4 eggs, each with a smooth, off-white shell, arranged in a row on the kitchen counter.\n- A large glass mixing bowl and various muffin ingredients (a bag of flour, a small bowl of sugar, a stick of butter, and a measuring cup of milk) are also on the counter.\n- Janet (white woman, same appearance as before) standing at the counter.\n- A speech bubble above Janet’s head stating: 'These will be great in muffins for my friends.' Composition: - The kitchen counter runs horizontally across the lower half of the scene.\n- The 4 eggs are placed in a neat row on the left side of the counter, each egg fully visible and not touching the others.\n- The mixing bowl is in the center of the counter, with the muffin ingredients grouped nearby but not overlapping.\n- Janet stands behind the counter, facing forward, with her hands positioned above the mixing bowl.\n- The speech bubble is above Janet’s head, with its tail pointing to her mouth, and does not overlap with Janet or the counter objects.\n- All objects mentioned in the scene are clearly visible and not overlapping; each is distinctly separated so that every object can be easily identified. Action: Janet cracks an egg into the mixing bowl, her gaze focused on the bowl. Scene 4: Objects: - A visually clear and attractive price list signboard with the textual label 'price list at farmers' market stall'.\n- The price list line1 states: 🥚  $2 / each\n- Janet (white woman, same appearance as before) standing behind a wooden farmers' market stall.\n- A speech bubble above Janet’s head stating: 'Fresh duck eggs for sale!' Composition: - The price list signboard is attached to the front of the stall, at eye level for customers, with the egg icon and price displayed on a single line, clearly legible.\n- Janet stands directly behind the stall, facing forward, with a basket of eggs visible on the stall counter.\n- The speech bubble is above Janet’s head, with its tail pointing to her mouth, and does not overlap with Janet or the price list.\n- All objects mentioned in the scene are clearly visible and not overlapping; each is distinctly separated so that every object can be easily identified. Action: Janet extends her right hand, pointing toward both the basket of eggs and the price list signboard, visually linking her announcement to the eggs and their price. --- FINAL SCENE 5(Problem Demonstration) --- Objects: - Janet (white woman, same appearance as before) standing at the farmers' market.\n- A large speech bubble above Janet’s head stating: 'How much do I make every day at the market?' Composition: - Janet stands at the center of the scene, facing forward.\n- The speech bubble is above Janet’s head, with its tail pointing directly to her mouth. The text is fully legible and the bubble does not overlap any other object.\n- All objects mentioned in the scene are clearly visible and not overlapping; each is distinctly separated so that every object can be easily identified. Action: Janet stands with her hands on her hips, looking thoughtfully ahead as she asks her question."

######################
Output:
Analysis:
Problem Identification:
Janet has eggs in a duck nest. She uses some for breakfast (Scene 2), some for baking (Scene 3),
and sells the rest at the farmers' market (Scene 4). 
We need to find out how much money she makes per day (Scene 5).

Scene 1 → Step 1 - Total eggs in nest:
There are 9 eggs visible in the nest.
Janet says there are 7 more eggs besides these.
Calculation: 9 + 7 = 16 eggs.

Scene 2 → Step 2 - Eggs left after breakfast:
Janet eats 3 eggs for breakfast.
Calculation: 16 - 3 = 13 eggs remain.

Scene 3 → Step 3 - Eggs left after baking muffins:
Janet uses 4 eggs to bake muffins.
Calculation: 13 - 4 = 9 eggs remain.

Scene 4 → Step 4 - Earnings from selling eggs:
Each egg sells for $2 at the farmers' market.
She sells the 9 remaining eggs.
Calculation: 9 * 2 = 18 dollars.

Scene 5 → Final Answer:
Janet makes $18 per day at the market.

FINAL ANSWER: 18


#####################



Example2:

Input: "Scene 1: Objects: - A visually clear and attractive price list with textual label 'price list'\n- The price list line1 states: 🏠  $80,000\n- Josh (white man, short brown hair, blue button-up shirt, khaki pants) stands next to the price list.\n- A speech bubble above Josh’s head stating: \"Here's what I paid for the house.\" Composition: - The price list is displayed on a wooden sign board staked into the ground in front of a house, with the icon 🏠 and '$80,000' on a single line, clearly visible and centered.\n- Josh stands to the right of the sign board, facing forward, with his right arm extended and index finger pointing directly at the price list.\n- The speech bubble is above Josh’s head, with the tail pointing to his mouth, and does not overlap the sign board or any other object.\n- All objects mentioned in the scene are clearly visible and not overlapping; each is distinctly separated so that every object can be easily identified. Action: Josh points to the house price sign with his right hand. Scene 2: Objects: - A visually clear and attractive sign board labeled 'on the kitchen counter'\n- The sign board line1 states: 🧾  $50,000\n- Josh (white man, same appearance as before) stands next to the kitchen counter and sign board.\n- A speech bubble above Josh’s head stating: \"That's what I spent on repairs.\" Composition: - The sign board is placed on the kitchen counter, with the icon 🧾 and '$50,000' on a single line, clearly visible and centered.\n- Josh stands to the right of the counter, facing forward, with his right arm extended and index finger pointing directly at the repair bill sign board.\n- The speech bubble is above Josh’s head, with the tail pointing to his mouth, and does not overlap the sign board or any other object.\n- All objects mentioned in the scene are clearly visible and not overlapping; each is distinctly separated so that every object can be easily identified. Action: Josh points to the repair bill sign board with his right hand. Scene 3: Objects: - A single circle visually displayed as a pie chart representing the ratio of green and gray segments.\n- The circle is evenly divided into 5 equal radial segments, starting at the top center and proceeding clockwise:\n  - The first 3 segments are filled with solid green (label: 60%).\n  - The next 2 segments are filled with solid gray (label: 40%).\n- Below the pie chart is a legend consisting of horizontal items:\n  - A small green square followed by the text 'increased value'.\n  - A small gray square followed by the text 'original value'.\n- Josh (white man, same appearance as before) stands next to the pie chart. Composition: - The pie chart is displayed on a presentation easel or board at the center of the scene, rendered in clean, high-contrast vector style.\n- The green wedge cluster is labeled '60%' in black text, centered over the green area; the gray wedge cluster is labeled '40%' in black text, centered over the gray area.\n- The legend is placed directly below the pie chart, with the green and gray squares and their respective text labels.\n- Josh stands to the right of the pie chart, facing forward, with his right arm extended and index finger pointing directly at the pie chart.\n- All objects mentioned in the scene are clearly visible and not overlapping; each is distinctly separated so that every object can be easily identified. Action: Josh points to the pie chart showing the increase. --- FINAL SCENE 4(Problem Demonstration) --- Objects: - Josh (white man, same appearance as before) is present in the scene.\n- A large speech bubble above Josh’s head stating: \"How much profit did I make?\" Composition: - Josh stands centrally in the scene, facing forward.\n- The speech bubble is above Josh’s head, with its tail pointing to Josh’s mouth. The text is fully legible and the bubble does not overlap any other object.\n- All objects mentioned in the scene are clearly visible and not overlapping; each is distinctly separated so that every object can be easily identified. Action: Josh looks forward, gaze focused, and the tail of the speech bubble points toward his mouth.",

Output:
Analysis:
Problem Identification:
We need to find Josh's profit. 
He bought the house (Scene 1), spent money on repairs (Scene 2), 
and the pie chart (Scene 3) shows that the increased value is 60% 
while the original value is 40%. 
This means the current price is in the ratio 5 : 2 compared to the original price. 
Finally, in Scene 4, Josh asks: "How much profit did I make?"

Scene 1 → Step 1 - Purchase price:
Purchase = 80,000.

Scene 2 → Step 2 - Repair cost:
Repairs = 50,000.

Scene 3 → Step 3 - Current price from ratio:
Original value corresponds to "2" parts.
Increased value corresponds to "3" parts.
So current value = (5 / 2) × 80,000 
= 2.5 × 80,000 
= 200,000.

Scene 4 → Step 4 - Profit calculation:
Total cost = 80,000 + 50,000 = 130,000
Profit = Current value - Total cost
= 200,000 - 130,000
= 70,000.

Final Answer:
Josh made a profit of $70,000.


"""

    return f"""You are an expert at analyzing visual scenes and solving mathematical problems based on interconnected information across multiple scenes.

{format_instruction}Analyze the following sequence of scenes carefully. Each scene provides information that connects to the others, building toward the final question. You need to understand the relationships between scenes and track quantities, actions, and information flow.

{example}

Now analyze these scenes:

{scene_description}

Please think step by step, connecting information across all scenes. After your reasoning, output your final answer on a new line starting with "FINAL ANSWER: " followed by the number only.
"""


def get_scene_extraction_prompt(scene_description: str, ground_truth: str) -> str:
    """
    Generate prompt for scene-based answer extraction with ground truth guidance.

    Args:
        scene_description: The complete scene description
        ground_truth: The expected ground truth answer

    Returns:
        Formatted prompt for answer extraction
    """
    return f"""You are analyzing a sequence of visual scenes to solve a math problem. The scenes contain interconnected information about quantities, actions, and relationships.

Scene Description:
{scene_description}

Expected Answer: {ground_truth}

Please extract the key information from each scene and show how they connect to arrive at the final answer.

FINAL ANSWER: """