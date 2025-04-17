# Sustainabite
[![Sustainabite Demo](https://github.com/jordyn-knock/Sustainabite/blob/main/logo.png)](https://youtu.be/uauizPwLQWU)
Click to see a demo!

-> How might we transform surplus and wasted food into a valuable resource for food-insecure communities?

## Our Focus
Target user -> Students, struggling with food waste and budgets while adapting to cooking for themselves!

Main goal -> Provide a tool to reduce overconsumption, promoting usage of what you already have to minimize food waste.

THUS, leading to the Sustainabite! Users are able to snap a quick image of their ingredients, pantry or refridgerator, fill out a few preferences on cuisine and cooking time and yield a few recipes to optimize the ingredients they have at home. 

## How we built it
We built a website based on a Kaggle dataset of 500,000 unique recipes.

IMAGE DETECTION
-> Integrating **CLIP by OpenAI**, the model understands images and text in conjunction, enabling identification
-> Pulls ingredients from a list of 500+ common ingredients
-> Fine tuned on the common data base

OVERARCHING AI MODEL (code name: botboclaat)
-> Trained on **retrieving a recipe** based on the following user inputs
    - Ingredients, scraped from image recognition
    - Cuisine, user input
    - Time spent for cooking, user input

CUISINE MODEL (code name: miss worldwide)
-> Trained on **predicting cuisine** from ingredients list
    - American, Caribbean, Chinese, French, German, Greek, Indian, Irish, Italian, Japanese, Korean, Mexican, Moroccan, Spanish, Thai
    & Vietnamese

MEAL MODEL (code name: balerinna cappuncinna, mimimimi)
-> Currently not in usage
    - When in usage minimized output a little too much
-> Trained on predicting meal type based on ingredients list

TIME MODEL (code name: pwincesssss!)
-> Trained on **identifying time measurements** based on tags

FRONTEND ON STREAMLIT
-> Simple front end hosted through **Streamlit**
-> User friendly and simple
-> Key features include:
    - Find recipe; prompts user for input & locates recipe based on dataset
    - Pantry cupboard; saves items in pantry or places them onto grocery lists
    - Favorites; a collection of saved recipes
