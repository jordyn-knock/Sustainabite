YOUCODE - SAP CASE - FOOD STASH

-> How might we transform surplus and wasted food into a valuable resource for food-insecure communities?

OUR FOCUS
Target user -> Students, struggling with food waste and budgets while adapting to cooking for themselves!

Main goal -> Provide a tool to reduce overconsumption, promoting usage of what you already have to minimize food waste.

THUS, leading to the Sustainabite! Users are able to snap a quick image of their ingredients, pantry or refridgerator, fill out a few preferences on cuisine and cooking time and yield a few recipes to optimize the ingredients they have at home. 

Based on a Kaggle dataset of 500 000 unique recipes, we built our mobile app. 

IMAGE DETECTION
-> integrating CLIP by OpenAI, the model understands images and text in conjunction, enabling identification
-> pulls ingredients from a list of 500+ common ingredients

OVERARCHING AI MODEL (code name: botboclaat)
-> trained on retrieving a recipe based on the following user inputs
    - ingredients, scraped from image recognition
    - cuisine, user input
    - time spent for cooking, user input

CUISINE MODEL (code name: miss worldwide)
-> trained on predicting cuisine from ingredients list
    - american, caribbean, chinese, french, german, greek, indian, irish, italian, japanese, korean, mexican, moroccan, spanish, thai
    & vietnamese

MEAL MODEL (code name: balerinna cappuncinna, mimimimi)
-> currently not in usage
    - when in usage minimized output a little too much
-> trained on predicting meal type based on ingredients list

TIME MODEL (code name: pwincesssss!)
-> trained on identifying time measurements based on tags

FRONTEND ON STREAMLIT
-> simple front end hosted through streamlit
-> user friendly and simple
-> key features include:
    - find recipe; prompts user for input & locates recipe based on dataset
    - pantry cupboard; saves items in pantry or places them onto grocery lists
    - favorites; a collection of saved recipes