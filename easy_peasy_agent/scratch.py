# from google.adk.agents.llm_agent import Agent

# root_agent = Agent(
#     model='gemini-2.5-flash-lite',
#     name='root_agent',
#     description='A helpful assistant for user questions.',
#     instruction='Answer user questions to the best of your knowledge',
# )
from dotenv import load_dotenv
import os

from google.adk.agents.callback_context import CallbackContext
from google.adk.agents import Agent, SequentialAgent, ParallelAgent, LoopAgent, LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.models import LlmResponse, LlmRequest
from google.adk.runners import InMemoryRunner, Runner
from google.adk.tools import AgentTool, FunctionTool, google_search
from google.genai import types
from google.adk.tools.tool_context import ToolContext
from google.adk.sessions import DatabaseSessionService, InMemorySessionService
import datetime

from pathlib import Path
import asyncio
from typing import Any, Dict, List, Optional
import traceback

import sqlite3


# Global
APP_NAME = "default"
USER_ID = "default-2"
SESSION = "default"

MODEL_NAME = "gemini-2.5-flash-lite"

ONBOARDING_FLAG_KEY = "user:number_of_meals"

ONBOARDING_COMPLETE_SIGNAL = "ONBOARDING_COMPLETE" 



# Setting some user preferences
LOCATION = "na"
NUM_RECIPES = 0

# Loading API Keys
load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

USER_NAME_SCOPE_LEVELS = ("temp", "user", "app")

#############################
# Creating memory functions
#############################

def save_user_info(
        tool_context: ToolContext,
        city: str,
        ingredients_to_avoid: List,
        personal_recipes: List,
        components_of_meal: str,
        number_of_meals: int
) -> Dict[str, Any]:
    """Tool that saves key information about the user to the session state

    Args:
        city (str): the city where the user does groceries
        ingredients_to_avoid (List): List of ingredients the user wants to avoid and never buy in the groceries
        personal_recipes (List): List of recipes the user likes
        components_of_meal (str): 3 or 4 items that are required in ever meal, for example: a protein, a vegetable and a starch.
        number_of_meals (int): Number of meals the user wants to cook each week

    Returns:
        Dict[str, Any]: dictionary returning status, whether success or failure.
        if failure, it is likely that one or more values couldn't be saved in the
        session state. Retrying could help.
    """
    tool_context.state["user:city"] = city
    tool_context.state["user:components_of_meal"] = components_of_meal
    tool_context.state["user:number_of_meals"] = number_of_meals
    # Initializing lists if not present, growing them if present
    if not ("user:ingredients_to_avoid" in tool_context.state):
        tool_context.state["user:ingredients_to_avoid"] = []
    tool_context.state["user:ingredients_to_avoid"] += ingredients_to_avoid
    if not ("user:personal_recipes" in tool_context.state):
        tool_context.state["user:personal_recipes"] = []
    tool_context.state["user:personal_recipes"] += personal_recipes

def retrieve_user_info(
        tool_context: ToolContext,
) -> Dict[str, Any]:
    """Tool that retrieves key information about the user from the session state

    Returns:
    Dict[str, Any]: 
        city (str): the city where the user does groceries
        ingredients_to_avoid (List): List of ingredients the user wants to avoid and never buy in the groceries
        personal_recipes (List): List of recipes the user likes
        components_of_meal (str): 3 or 4 items that are required in ever meal, for example: a protein, a vegetable and a starch.
        number_of_meals (int): Number of meals the user wants to cook each week
    """
    user_dict = {}
    user_dict["city"] = tool_context.state["user:city"]
    user_dict["components_of_meal"] = tool_context.state.get("user:components_of_meal", "Components of meal not found")
    user_dict["number_of_meals"] = tool_context.state.get("user:number_of_meals", "Number of meals not found")
    user_dict["ingredients_to_avoid"] = tool_context.state.get("user:ingredients_to_avoid", "Ingredients_to_avoid not found")
    user_dict["personal_recipes"] = tool_context.state.get("user:personal_recipes", "personal_recipes not found")
    return user_dict

   
def save_city(
    tool_context: ToolContext, city: str
) -> Dict[str, Any]:
    """
    Tool to record and save the city where the user lives.

    Args:
        city: The name of the user's city
    """
    # Write to session state using the 'user:' prefix for user data
    tool_context.state["user:city"] = city
    return {"status": "success", "city": city}

def retrieve_city(tool_context: ToolContext) -> Dict[str, Any]:
    """
    Tool to retrieve the city where the user lives.
    """
    # Read from session state
    city = tool_context.state.get("user:city", "City not found")

    return {"status": "success", "city": city}

def save_ingredient_to_avoid(
        tool_context: ToolContext, ingredient_to_avoid: str
) -> Dict[str, Any]:
    """
    Tool to record and save ingredients that the user wants to avoid or exclude

    Args:
        ingredient_to_avoid: The name of the ingredient to avoid or exclude
    """
    if not ("user:ingredients_to_avoid" in tool_context.state):
        tool_context.state["user:ingredients_to_avoid"] = []
    tool_context.state["user:ingredients_to_avoid"].append(ingredient_to_avoid)
    return {"status": "success", "ingredient_to_avoid":ingredient_to_avoid}

def retrieve_ingredients_to_avoid(
        tool_context: ToolContext
                                  ) -> Dict[str, Any]:
    """
    Tool to retrieve the list of ingredients that the user must avoid or exclude.
    """
    # Read from session state
    list_of_ingredients_to_avoid = tool_context.state.get("user:ingredients_to_avoid", "No ingredients to avoid found")

    return {"status": "success", "city": list_of_ingredients_to_avoid}

def save_personal_recipes(
        tool_context: ToolContext, recipe: str
) -> Dict[str, Any]:
    """
    Tool to record and save any recipes that the user wants

    Args:
        recipe: The recipe
    """
    if not ("user:personal_recipes" in tool_context.state):
        tool_context.state["user:personal_recipes"] = []
    tool_context.state["user:personal_recipes"].append(recipe)
    return {"status": "success", "personal_recipe":recipe}

def save_components_of_meal(
        tool_context: ToolContext, components: str
) -> Dict[str, Any]:
    """
    Tool to record and save the main components of a meal

    Args:
        components: The components that make a good meal, such as: protein, vegetables and grains
    """
    tool_context.state["user:meal_components"] = components
    return {"status": "success", "meal_components":components}

def retrieve_components_of_a_meal(tool_context: ToolContext
                                  ) -> Dict[str, Any]:
    """
    Tool to retrieve the list of components that every meal must have
    """
    # Read from session state
    meal_components = tool_context.state.get("user:meal_components", "No components found, to be safe, use protein, vegetables and grains, double check with user")

    return {"status": "success", "meal_components": meal_components}

def save_number_of_meals(
        tool_context: ToolContext, number_of_meals: str
) -> Dict[str, Any]:
    """
    Tool to record and save how many meals the user wants to cook per week

    Args:
        number_of_meals: The number of meals the user wants to cook per week"""
    tool_context.state["user:number_of_meals"] = number_of_meals
    return {"status": "success", "number_of_meals":number_of_meals}

def retrieve_number_of_meals(tool_context: ToolContext
                                  ) -> Dict[str, Any]:
    """
    Tool to retrieve the number of meals the user wants to cook per week
    """
    # Read from session state
    number_of_meals = tool_context.state.get("user:number_of_meals", "No value found, use 3, double check with user")

    return {"status": "success", "meal_components": number_of_meals}

def state_checker(tool_context: ToolContext
                                  ) -> Dict[str, Any]:
    """
    Tool to make sure onboarding completed smoothly and doesn't need to run again
    Returns: dict "onboarding_needed"
    Boolean, if true, onboarding questions are needed to complete the questionnaire
    if false, the user is all set with the minimum state variables (city, number of meals and meal components)
    """
    for key in ["number_of_meals","meal_components", "city"]: 
        if not ("user:"+key in tool_context.state):
            return {"onboarding_needed": True}
    return {"onboarding_needed": False}
#####################


############################
# Creating Callback functions
###########################
def skip_onboarding_if_complete(
    callback_context: CallbackContext,
    llm_request:LlmRequest 
) -> Optional[types.Content]:
    """
    Checks the session state for a completion flag and returns a Content object 
    to signal the ADK runtime to skip agent execution.
    """
    
    # 1. Access the session state via the callback_context
    state = callback_context.session.state
    
    # 2. Check the persistent flag
    if ONBOARDING_FLAG_KEY in state:
        
        # print(f"[{callback_context.agent_name}] Callback-here: Skipping: Onboarding completed")
        
        # 3. Return a types.Content object to signal a skip.
        # This acts as the final result for the current agent execution, 
        # allowing the SequentialAgent to move to the next item in the sequence.
        return LlmResponse(
            content=types.Content(
                parts=[types.Part(text="__ONBOARDING_SKIP__")]
            )
        )
        
    # If the flag is not set, return None to proceed with the agent's main execution.
    return None


#######################
# Agent tools
######################


def get_current_month() -> dict:
    """
    Returns the current month. 
    Use this function to find what ingredients are in season now
    """
    # Get the current date (and time)
    now = datetime.datetime.now()
    
    # Format and return the date
    return {
        "current_month": now.strftime("%B")
        # "current_date": now.strftime("%Y-%m-%d"),
        # "timestamp": now.isoformat(),
    }

def exit_loop():
    """Call this tool after you finished asking all your questions, 
    getting the responses, and saving them using your tools"""
    return ONBOARDING_COMPLETE_SIGNAL



###########################
# Internal tools and configs
#############################
# Creating retry config
retry_config=types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1, # Initial delay before first retry (in seconds)
    http_status_codes=[429, 500, 503, 504] # Retry on these HTTP errors
)



# Function to run sessions
async def run_session(
    runner_instance: Runner,
    user_queries: list[str] | str = None,
    session_name: str = "default",
):
    print(f"\n ### Session: {session_name}")

    # Get app name from the Runner
    app_name = runner_instance.app_name

    # Attempt to create a new session or retrieve an existing one
    try:
        session = await session_service.create_session(
            app_name=app_name, user_id=USER_ID, session_id=session_name
        )
    except:
        session = await session_service.get_session(
            app_name=app_name, user_id=USER_ID, session_id=session_name
        )

    # Process queries if provided
    if user_queries:
        # Convert single query to list for uniform processing
        if type(user_queries) == str:
            user_queries = [user_queries]

        # Process each query in the list sequentially
        for query in user_queries:
            print(f"\nUser > {query}")

            # Convert the query string to the ADK Content format
            query = types.Content(role="user", parts=[types.Part(text=query)])

            # Stream the agent's response asynchronously
            async for event in runner_instance.run_async(
                user_id=USER_ID, session_id=session.id, new_message=query
            ):
                # Check if the event contains valid content
                if event.content and event.content.parts:
                    # Filter out empty or "None" responses before printing
                    if (
                        event.content.parts[0].text != "None"
                        and event.content.parts[0].text
                    ):
                        print(f"{MODEL_NAME} > ", event.content.parts[0].text)
    else:
        print("No queries!")

######################
##### Agents #########
########################
# Ingredients Agent: Its job is to use the google_search tool to list ingredients that are in season.
ingredients_in_season_agent = Agent(
    name="IngredientsAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction="""You are a specialized research agent. Your only job is to use the
    google_search tool to find 5-6 ingredients in season for {user:city}. 
    Return only the list of 5-6 ingredients and nothing else""",
    tools=[google_search],
    output_key="ingredients_list",  # The result of this agent will be stored in the session state with this key.
)

# instruction=f"""You are an agent in a team of grocery assistants. Your 
#     job is to find suitable ingredients in season and put it in a clear
#     output that another agent can understand.
#     To do that, follow these steps:
#     1. Retrieve the city the user is located using the `retrieve_city` tool
#     2. Retrieve the current month using the `get_current_month` tool
#     3. Get a list of vegetables and ingredients in season for the current month and city
#     4. Use `retrieve_ingredients_to_avoid` tool to find which ingredients should avoid
#     5. Return a list of vegetable and ingredients in season that does not contain any ingredients listed in ingredients_to_avoid""",
#     tools=[google_search,
#       FunctionTool(get_current_month), 
#         retrieve_city, retrieve_ingredients_to_avoid,],


# """You are a specialized research agent. Your only job is to use the
#     google_search tool to find 5-6 ingredients in season for the location {LOCATION}. 
#     Return only the list of ingredients and nothing else"""

# Recipe Agent: Its job is to find recipes that contain the ingredients it recieves
recipe_agent = Agent(
    name="RecipeAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction="""Use the ingredient list: {ingredients_list} and the meal
    components {user:meal_components} to find     {user:number_of_meals} 
    recipes that contain at least one of the ingredients
    found in the ingredient list. Make sure each recipe contains each element of
    {user:meal_components}. Return only the list of recipes and nothing else""",
    output_key="recipe_list",
    tools=[google_search],
)

# Root Coordinator: Orchestrates the workflow by calling the sub-agents as tools.
# simple_root_agent = Agent(
#     name="GroceryAssistant",
#     model=Gemini(
#         model="gemini-2.5-flash-lite",
#         retry_options=retry_config
#     ),
#     # This instruction tells the root agent HOW to use its tools (which are the other agents).
#     instruction=""""You are a helpful assistant. Your goal is to plan the groceries 
#     for the week. First you will call ingredients_in_season_agent to get a 
#     list of ingredients, and then you will call a recipe_agent who will create 
#     a list of recipes. You will then display that list of recipes to the user and ask for approval""",
#     # We wrap the sub-agents in `AgentTool` to make them callable tools for the root agent.
#     tools=[AgentTool(ingredients_in_season_agent), AgentTool(recipe_agent)],
# )


root_agent = LlmAgent(
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    name="OnboardingAgent",
    instruction="""You are an onboarding concierge of an app that helps people with their groceries. 
    Your goal is to get to know the user and save key information about them in your memory,
    using the tools below. 
    0. Welcome them to Easy Peasy Onboarding!
    1. Ask them in what city they do their groceries and wait for their response
    2. Call the `save_city` tool to save their response
    3. Ask them about ingredients to avoid, and tell them they can update this information at any time
    4. If some ingredients are mentioned, call the `save_ingredient_to_avoid` tool to save the ingredient(s)
    5. Ask them about any favorite recipes and reasure them they can provide those at any time
    6. If some recipes are mentioned, call the `save_personal_recipes` tool to save the recipe(s)
    7. Ask them what components make a complete meal, offer an example, such as: protein, vegetables and grains
    8. Call `save_components_of_meal` tool to save the user response or the example if the user accepted it
    9. Ask them how many meals do they plan to cook per week, assure them they can provide this information at any time
    10.If you get a response, call `save_number_of_meals` tool to save the user response
    11.Use `retrieve_user_info` tool to load te user dictionary and return its values.

    Tools for managing user context:
    * To record city when provided use `save_city` tool. 
    * To record ingredients to avoid when provided use `save_ingredient_to_avoid` tool. 
    * To record personal recipes when provided use `save_personal_recipes` tool. 
    * To record the components of a meal when provided or accpeted use `save_components_of_meal` tool. 

    """,
    # before_model_callback=skip_onboarding_if_complete,
    tools=[save_city, save_ingredient_to_avoid, save_personal_recipes, 
           save_components_of_meal, save_number_of_meals, retrieve_user_info], 
    output_key="user_dict"
)

onboarding_complete_agent = LlmAgent(
    model=Gemini(model=MODEL_NAME, retry_options=retry_config),
    name="OnboardCompleteAgent",
    instruction=""" Use `state_checker` tool to check if onboarding is still needed.
    If onboarding is not needed, call `exit_loop` tool and respond only
    with the function call and nothing else. 
    """,
    tools=[FunctionTool(exit_loop), FunctionTool(state_checker)]

)

# onboarding_loop = LoopAgent(
#     name="OnboardingLoop",
#     sub_agents=[onboarding_agent, onboarding_complete_agent],
#     max_iterations=6,  # Prevents infinite loops
# )

# root_agent = SequentialAgent(
#     name="Groceries_pipeline",
#     sub_agents=[onboarding_loop, ingredients_in_season_agent, recipe_agent]

# )

# root_agent = LlmAgent(
#     model=Gemini(model=MODEL_NAME, retry_options=retry_config),
#     name="EasyPeasyAgent",
#     instruction="""
#                     You are a helpful agent that will help users do their groceries
#                     First you need to use the state_checker tool to see if you need to
#                     complete oboarding. If `onboarding_needed==True` then 
#                     ask the user about what city they do groceries, how many 
#                     meals they want to cook a week, etc, and save this information
#                     using the save_user_info tool. 
#                     Once you call state_checker and it returns `onboarding_needed==False`
#                     Then you can move ahead and get the ingredients in season with
#                     ingredients_in_season_agent tool and then get the recipes
#                     with recipe_agent tool. 

#                 """,
#     tools=[FunctionTool(save_user_info), FunctionTool(retrieve_user_info), 
#            FunctionTool(state_checker),
#             AgentTool(ingredients_in_season_agent),
#             AgentTool(recipe_agent)]

# )
        #    FunctionTool(exit_loop)],  # Provide the tools to the agent


# onboarding_loop = LoopAgent(
#     name="OnboardingLoop",
#     sub_agents=[onboarding_agent],
#     max_iterations=6,  # Prevents infinite loops
# )

# orchestrator_agent = LlmAgent(
#     name="easy_peasy_agent",
#     model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
#     instruction="""You are a smart grocery assistant. Your job is to help the 
#     user with groceries by following the steps below:
#     1.Use tool `retrieve_city` to find where the user lives.
#      1.a. if City not found, present the user with the
#     onboarding_agent sub_agent. 

#     2. Once that's done, Use the ingredients_in_season_agent tool to generate a list of 
#     ingredients in season. Only call this tool ONCE without prompt.

#     3. Present the list to the user

#     """,
#     tools=[retrieve_city, AgentTool(ingredients_in_season_agent)],
#     sub_agents=[onboarding_agent]

# )

# recipe_pipeline = SequentialAgent(
#     name="RecipePipeline",
#     sub_agents=[ingredients_in_season_agent, recipe_agent]
# )


# sequential_root_agent = SequentialAgent(
#     name="RecipePipeline",
#     sub_agents=[onboarding_loop, ingredients_in_season_agent, recipe_agent]
# )


# some_agent = LlmAgent(
#     name="some_agent",
#     model="gemini-2.5-flash-lite",
#     instruction=(
#         "You are the easy-peasy grocery assistant.  Your job is to help the user"
#         "plan their groceries using the tools in the order below."
#         "O. Welcome the user to easy-peasy groceries!"
#         "1. ** Check state:** call state_checker tool, if it returns onboarding_needed = True"
#         "1a. **Onboard new user:**  call the `onboarding_loop`. tool."
#         "** Make sure to call `onboarding_loop` tool,  DON'T run some random onboarding by yourself. "
#         "2. **Get Ingredients:** If the previous tool result was onboarding_needed = False"
#         "3. **Get recipe:** Call the `recipe_agent` to get a list of recipes."
#         "4. **Confirm with the user:** Show recipes to the user and ask them if they want to change anything"
#         "You MUST only proceed to the next step if the previous step's data is available or completed."
#     ),
#     tools=[
#         AgentTool(onboarding_loop),
#         AgentTool(ingredients_in_season_agent),
#         AgentTool(recipe_agent),
#         FunctionTool(state_checker)
#     ]
# )
