# Helpful function to explore a database at the end
# Plus simple run_debug session example

from dotenv import load_dotenv
import os


from google.adk.agents import Agent, SequentialAgent, ParallelAgent, LoopAgent, LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner, Runner
from google.adk.tools import AgentTool, FunctionTool, google_search
from google.genai import types
from google.adk.tools.tool_context import ToolContext
from google.adk.sessions import DatabaseSessionService, InMemorySessionService

from pathlib import Path
import asyncio
from typing import Any, Dict, List
import traceback

import sqlite3


# Global
APP_NAME = "default"
USER_ID = "default"
SESSION = "default"

MODEL_NAME = "gemini-2.5-flash-lite"

# Setting some user preferences
LOCATION = "Boston"
NUM_RECIPES = 4

# Loading API Keys
load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

# Creating Memory to save user and session state with local database
db_url = 'sqlite+aiosqlite:///test_my_database.db'
try:
    session_service = DatabaseSessionService(db_url=db_url)
except Exception as e:
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {str(e)}")
    print("\nFull traceback:")
    traceback.print_exc()

USER_NAME_SCOPE_LEVELS = ("temp", "user", "app")

# Creating memory functions
def save_userinfo(
    tool_context: ToolContext, user_name: str, city: str
) -> Dict[str, Any]:
    """
    Tool to record and save user name and city in session state.

    Args:
        user_name: The username to store in session state
        country: The name of the user's country
    """
    # Write to session state using the 'user:' prefix for user data
    tool_context.state["user:name"] = user_name
    tool_context.state["user:city"] = city


def retrieve_userinfo(tool_context: ToolContext) -> Dict[str, Any]:
    """
    Tool to retrieve user name and city from session state.
    """
    # Read from session state
    user_name = tool_context.state.get("user:name", "Username not found")
    city = tool_context.state.get("user:city", "City not found")

    return {"status": "success", "user_name": user_name, "city": city}


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


# root_agent = Agent(
#     name="helpful_assistant",
#     model=Gemini(
#         model="gemini-2.5-flash-lite",
#         retry_options=retry_config
#     ),
#     description="A simple agent that can answer general questions.",
#     instruction="You are a helpful assistant. Use Google Search for current info or if unsure.",
#     tools=[google_search],
# )




# Ingredients Agent: Its job is to use the google_search tool to list ingredients that are in season.
ingredients_in_season_agent = Agent(
    name="IngredientsAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction=f"""You are a specialized research agent. Your only job is to use the
    google_search tool to find 5-6 ingredients in season for the location {LOCATION}. 
    Return only the list of ingredients and nothing else""",
    tools=[google_search],
    output_key="ingredients_list",  # The result of this agent will be stored in the session state with this key.
)



# Recipe Agent: Its job is to find recipes that contain the ingredients it recieves
recipe_agent = Agent(
    name="RecipeAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    # The instruction is modified to request a bulleted list for a clear output format.
    instruction="""Use the ingredient list: {ingredients_list}""" +
    f""""to find {NUM_RECIPES} that contain at least one of the ingredients
    found in the ingredient list. Return only the list of recipes and nothing else""",
    output_key="recipe_list",
    tools=[google_search],
)

# Root Coordinator: Orchestrates the workflow by calling the sub-agents as tools.
simple_root_agent = Agent(
    name="GroceryAssistant",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    # This instruction tells the root agent HOW to use its tools (which are the other agents).
    instruction=""""You are a helpful assistant. Your goal is to plan the groceries 
    for the week. First you will call ingredients_in_season_agent to get a 
    list of ingredients, and then you will call a recipe_agent who will create 
    a list of recipes. You will then display that list of recipes to the user and ask for approval""",
    # We wrap the sub-agents in `AgentTool` to make them callable tools for the root agent.
    tools=[AgentTool(ingredients_in_season_agent), AgentTool(recipe_agent)],
)

sequential_root_agent = SequentialAgent(
    name="RecipePipeline",
    sub_agents=[ingredients_in_season_agent, recipe_agent],
)

test_root_agent = LlmAgent(
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    name="text_chat_bot",
    description="""A text chatbot.
    Tools for managing user context:
    * To record username and city when provided use `save_userinfo` tool. 
    * To fetch username and city when required use `retrieve_userinfo` tool.
    """,
    tools=[save_userinfo, retrieve_userinfo],  # Provide the tools to the agent
)

session_service = InMemorySessionService()


runner = Runner(agent=ingredients_in_season_agent, session_service=session_service, app_name=APP_NAME)

# async def main():
#     await run_session(
#         runner,
#         [
#             "Hi there, how are you doing today? What is my name?",  # Agent shouldn't know the name yet
#             "My name is Sam. I'm from Boston.",  # Provide name - agent should save it
#             "What is my name? Which country am I from?",  # Agent should recall from session state
#         ],
#         "state-demo-session",
#     )

async def main():
    try:  # run_debug() requires ADK Python 1.18 or higher:
        response = await runner.run_debug(
                    "What's the ingredients list", verbose=True,
                    session_id="Test_session")
        
        session = await session_service.get_session(
                        app_name=APP_NAME, user_id=USER_ID, 
                        session_id="Test_session")
        if session:
            print(session.state)


    except Exception as e:
        print(f"An error occurred during agent execution: {e}")
        print("\nFull traceback:")
        traceback.print_exc()
    




asyncio.run(main())


# Check database
def check_data_in_db(): 
    with sqlite3.connect("test_my_database.db") as connection:
        cursor = connection.cursor()
        result = cursor.execute(
            "select app_name, session_id, author, content from events"
        )
        print([_[0] for _ in result.description])
        for each in result.fetchall():
            print(each)


# check_data_in_db()


async def debug_sessions():
    # List all sessions to see what's in the database
    response = await session_service.list_sessions(
        app_name=APP_NAME,
        user_id=USER_ID
    )
    
    # Access the sessions from the response object
    print(f"Response type: {type(response)}")
    print(f"Response: {response}")
    
    # Try to access sessions - could be .sessions or .items
    if hasattr(response, 'sessions'):
        sessions = response.sessions
        print(f"\nFound {len(sessions)} sessions:")
        for s in sessions:
            print(f"    State: {s.state}")
    elif hasattr(response, 'items'):
        sessions = response.items
        print(f"\nFound {len(sessions)} sessions:")
        for s in sessions:
            print(f"    State: {s.state}")
    else:
        print(f"\nAttributes available: {dir(response)}")

# asyncio.run(debug_sessions())


async def main():
    try:
        # Step 1: Run the agent
        print("Running agent...")
        response = await runner.run_debug(
            "My name is mavi and my city is boston", 
            verbose=True,
            session_id="Test_session"
        )
        print(f"Response: {response}\n")
        
        # Step 2: Get the session
        print("Fetching session...")
        session = await session_service.get_session(
            app_name=APP_NAME, 
            user_id=USER_ID, 
            session_id="Test_session"
        )
        
        # Step 3: Debug the session
        print(f"Session object: {session}")
        print(f"Session is None: {session is None}")
        
        if session:
            print(f"Session state: {session.state}")
            print(f"Session ID: {session.session_id}")
            # Print all session attributes to see what's available
            print(f"Session attributes: {dir(session)}")
        else:
            print("⚠️ Session is None - it wasn't created or found")
            
    except Exception as e:
        print(f"An error occurred: {e}")
        print("\nFull traceback:")
        traceback.print_exc()

# Run it
# asyncio.run(main())


# Configuration
APP_NAME = "default"
USER_ID = "default"
MODEL_NAME = "gemini-2.5-flash-lite"

# Create an agent with session state tools
root_agent = LlmAgent(
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    name="text_chat_bot",
    description="""A text chatbot.
    Tools for managing user context:
    * To record username and city when provided use `save_userinfo` tool. 
    * To fetch username and city when required use `retrieve_userinfo` tool.
    """,
    tools=[save_userinfo, retrieve_userinfo],  # Provide the tools to the agent
)

# Set up session service and runner
# session_service = InMemorySessionService()
# Creating Memory to save user and session state with local database
db_url = 'sqlite+aiosqlite:///test_my_database.db'
# session_service = DatabaseSessionService(db_url=db_url)

# runner = Runner(agent=root_agent, session_service=session_service, app_name="default")


async def main():
    try:  # run_debug() requires ADK Python 1.18 or higher:
        await run_session(
            runner,
            [
                "Hi there, how are you doing today? What is my name?",  # Agent shouldn't know the name yet
                "My name is Sam. I'm from Poland.",  # Provide name - agent should save it
                "What is my name? Which country am I from?",  # Agent should recall from session state
            ],
            "state-demo-session",
        )

        # Retrieve the session and inspect its state
        session = await session_service.get_session(
            app_name=APP_NAME, user_id=USER_ID, session_id="state-demo-session"
        )

        print("Session State Contents:")
        print(session.state)



    except Exception as e:
        print(f"An error occurred during agent execution: {e}")
        print("\nFull traceback:")
        traceback.print_exc()
    



async def main():
    try:  # run_debug() requires ADK Python 1.18 or higher:
        response = await runner.run_debug("What's the ingredients list?", 
                                          session_id="state-demo-session",
                                          verbose=True)
        

        # Retrieve the session and inspect its state
        session = await session_service.get_session(
            app_name=APP_NAME, user_id=USER_ID, session_id="state-demo-session"
        )

        print("Session State Contents:")
        print(session.state)



    except Exception as e:
        print(f"An error occurred during agent execution: {e}")
        # print("\nFull traceback:")
        traceback.print_exc()


# asyncio.run(main())

import sqlite3

# Super helpful function to explore a database, by Claude
def explore_database(db_path="test_my_database.db"):
    """
    Comprehensive function to explore a SQLite database
    """
    with sqlite3.connect(db_path) as connection:
        cursor = connection.cursor()
        
        print("=" * 60)
        print(f"EXPLORING DATABASE: {db_path}")
        print("=" * 60)
        
        # 1. List all tables in the database
        print("\n📋 TABLES IN DATABASE:")
        print("-" * 60)
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' 
            ORDER BY name;
        """)
        tables = cursor.fetchall()
        
        if not tables:
            print("No tables found in database!")
            return
        
        for (table_name,) in tables:
            print(f"  • {table_name}")
        
        # 2. For each table, show its structure and sample data
        for (table_name,) in tables:
            print("\n" + "=" * 60)
            print(f"TABLE: {table_name}")
            print("=" * 60)
            
            # Get column information
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()
            
            print("\n📊 COLUMNS:")
            print("-" * 60)
            print(f"{'Column Name':<20} {'Type':<15} {'NotNull':<10} {'Default':<15} {'PK':<5}")
            print("-" * 60)
            for col in columns:
                cid, name, col_type, not_null, default_val, pk = col
                print(f"{name:<20} {col_type:<15} {not_null:<10} {str(default_val):<15} {pk:<5}")
            
            # Count rows
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            row_count = cursor.fetchone()[0]
            print(f"\n📈 Total rows: {row_count}")
            
            # Show sample data (first 5 rows)
            if row_count > 0:
                print("\n🔍 SAMPLE DATA (first 5 rows):")
                print("-" * 60)
                cursor.execute(f"SELECT * FROM {table_name} LIMIT 5")
                
                # Get column names for header
                col_names = [description[0] for description in cursor.description]
                print(" | ".join(col_names))
                print("-" * 60)
                
                for row in cursor.fetchall():
                    # Truncate long values for display
                    formatted_row = []
                    for value in row:
                        if value is None:
                            formatted_row.append("NULL")
                        elif isinstance(value, str) and len(value) > 50:
                            formatted_row.append(value[:47] + "...")
                        else:
                            formatted_row.append(str(value))
                    print(" | ".join(formatted_row))
            else:
                print("(Table is empty)")
        
        print("\n" + "=" * 60)
        print("EXPLORATION COMPLETE")
        print("=" * 60)

# explore_database("easy_peasy_database.db")
