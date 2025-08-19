import os
import sqlite3
import requests
import ast
import uuid
from typing import List, Optional, Annotated
from typing_extensions import TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.utilities.sql_database import SQLDatabase
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool
from pydantic import BaseModel, Field

from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.managed.is_last_step import RemainingSteps
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, create_react_agent
from langgraph.types import interrupt, Command
from langgraph.store.base import BaseStore

from langchain_core.tools import tool
from langchain_core.messages import ToolMessage, SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

# Load environment variables
load_dotenv(dotenv_path=".env", override=True)

# Initialize the ChatOpenAI model
llm = ChatOpenAI(model="gpt-4o", temperature=0)

def get_engine_for_chinook_db():
    """Pull sql file, populate in-memory database, and create engine."""
    url = "https://raw.githubusercontent.com/lerocha/chinook-database/master/ChinookDatabase/DataSources/Chinook_Sqlite.sql"
    
    response = requests.get(url)
    sql_script = response.text

    connection = sqlite3.connect(":memory:", check_same_thread=False)
    connection.executescript(sql_script)
    
    return create_engine(
        "sqlite://",
        creator=lambda: connection,
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )

# Get the SQLAlchemy engine for our Chinook database
engine = get_engine_for_chinook_db()
db = SQLDatabase(engine)

# Initialize memory systems
in_memory_store = InMemoryStore()
checkpointer = MemorySaver()

# Define the state schema
class State(TypedDict):
    """Represents the state of our LangGraph agent."""
    customer_id: str
    messages: Annotated[list[AnyMessage], add_messages]
    loaded_memory: str
    remaining_steps: RemainingSteps

# Music tools
@tool
def get_albums_by_artist(artist: str):
    """Get albums by an artist."""
    return db.run(
        f"""
        SELECT Album.Title, Artist.Name 
        FROM Album 
        JOIN Artist ON Album.ArtistId = Artist.ArtistId 
        WHERE Artist.Name LIKE '%{artist}%';
        """,
        include_columns=True
    )

@tool
def get_tracks_by_artist(artist: str):
    """Get songs by an artist (or similar artists)."""
    return db.run(
        f"""
        SELECT Track.Name as SongName, Artist.Name as ArtistName 
        FROM Album 
        LEFT JOIN Artist ON Album.ArtistId = Artist.ArtistId 
        LEFT JOIN Track ON Track.AlbumId = Album.AlbumId 
        WHERE Artist.Name LIKE '%{artist}%';
        """,
        include_columns=True
    )

@tool
def get_songs_by_genre(genre: str):
    """Fetch songs from the database that match a specific genre."""
    genre_id_query = f"SELECT GenreId FROM Genre WHERE Name LIKE '%{genre}%'"
    genre_ids = db.run(genre_id_query)
    
    if not genre_ids:
        return f"No songs found for the genre: {genre}"
    
    genre_ids = ast.literal_eval(genre_ids)
    genre_id_list = ", ".join(str(gid[0]) for gid in genre_ids)

    songs_query = f"""
        SELECT Track.Name as SongName, Artist.Name as ArtistName
        FROM Track
        LEFT JOIN Album ON Track.AlbumId = Album.AlbumId
        LEFT JOIN Artist ON Album.ArtistId = Artist.ArtistId
        WHERE Track.GenreId IN ({genre_id_list})
        GROUP BY Artist.Name
        LIMIT 8;
    """
    songs = db.run(songs_query, include_columns=True)
    
    if not songs:
        return f"No songs found for the genre: {genre}"
        
    formatted_songs = ast.literal_eval(songs)
    return [
        {"Song": song["SongName"], "Artist": song["ArtistName"]}
        for song in formatted_songs
    ]

@tool
def check_for_songs(song_title):
    """Check if a song exists by its name."""
    return db.run(
        f"""
        SELECT * FROM Track WHERE Name LIKE '%{song_title}%';
        """,
        include_columns=True
    )

music_tools = [get_albums_by_artist, get_tracks_by_artist, get_songs_by_genre, check_for_songs]
llm_with_music_tools = llm.bind_tools(music_tools)
music_tool_node = ToolNode(music_tools)

# Music assistant
def generate_music_assistant_prompt(memory: str = "None") -> str:
    return f"""
    You are a member of the assistant team, your role specifically is to focused on helping customers discover and learn about music in our digital catalog. 
    If you are unable to find playlists, songs, or albums associated with an artist, it is okay. 
    Just inform the customer that the catalog does not have any playlists, songs, or albums associated with that artist.
    You also have context on any saved user preferences, helping you to tailor your response. 
    
    CORE RESPONSIBILITIES:
    - Search and provide accurate information about songs, albums, artists, and playlists
    - Offer relevant recommendations based on customer interests
    - Handle music-related queries with attention to detail
    - Help customers discover new music they might enjoy
    - You are routed only when there are questions related to music catalog; ignore other questions. 
    
    SEARCH GUIDELINES:
    1. Always perform thorough searches before concluding something is unavailable
    2. If exact matches aren't found, try:
       - Checking for alternative spellings
       - Looking for similar artist names
       - Searching by partial matches
       - Checking different versions/remixes
    3. When providing song lists:
       - Include the artist name with each song
       - Mention the album when relevant
       - Note if it's part of any playlists
       - Indicate if there are multiple versions
    
    Additional context is provided below: 

    Prior saved user preferences: {memory}
    
    Message history is also attached.  
    """

def music_assistant(state: State, config: RunnableConfig): 
    memory = "None" 
    if "loaded_memory" in state: 
        memory = state["loaded_memory"]

    music_assistant_prompt = generate_music_assistant_prompt(memory)
    response = llm_with_music_tools.invoke([SystemMessage(music_assistant_prompt)] + state["messages"])
    
    return {"messages": [response]}

def should_continue(state: State, config: RunnableConfig):
    messages = state["messages"]
    last_message = messages[-1]
    
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"

# Create music catalog subagent
music_workflow = StateGraph(State)
music_workflow.add_node("music_assistant", music_assistant)
music_workflow.add_node("music_tool_node", music_tool_node)
music_workflow.add_edge(START, "music_assistant")
music_workflow.add_conditional_edges(
    "music_assistant",
    should_continue,
    {
        "continue": "music_tool_node",
        "end": END,
    },
)
music_workflow.add_edge("music_tool_node", "music_assistant")

music_catalog_subagent = music_workflow.compile(
    name="music_catalog_subagent", 
    checkpointer=checkpointer, 
    store=in_memory_store
)

# Invoice tools
@tool 
def get_invoices_by_customer_sorted_by_date(customer_id: str) -> list:
    """Look up all invoices for a customer using their ID."""
    return db.run(f"SELECT * FROM Invoice WHERE CustomerId = {customer_id} ORDER BY InvoiceDate DESC;")

@tool 
def get_invoices_sorted_by_unit_price(customer_id: str) -> list:
    """Use this tool when the customer wants to know the details of one of their invoices based on the unit price/cost of the invoice."""
    query = f"""
        SELECT Invoice.*, InvoiceLine.UnitPrice
        FROM Invoice
        JOIN InvoiceLine ON Invoice.InvoiceId = InvoiceLine.InvoiceId
        WHERE Invoice.CustomerId = {customer_id}
        ORDER BY InvoiceLine.UnitPrice DESC;
    """
    return db.run(query)

@tool
def get_employee_by_invoice_and_customer(invoice_id: str, customer_id: str) -> dict:
    """This tool will take in an invoice ID and a customer ID and return the employee information associated with the invoice."""
    query = f"""
        SELECT Employee.FirstName, Employee.Title, Employee.Email
        FROM Employee
        JOIN Customer ON Customer.SupportRepId = Employee.EmployeeId
        JOIN Invoice ON Invoice.CustomerId = Customer.CustomerId
        WHERE Invoice.InvoiceId = ({invoice_id}) AND Invoice.CustomerId = ({customer_id});
    """
    
    employee_info = db.run(query, include_columns=True)
    
    if not employee_info:
        return f"No employee found for invoice ID {invoice_id} and customer identifier {customer_id}."
    return employee_info

invoice_tools = [get_invoices_by_customer_sorted_by_date, get_invoices_sorted_by_unit_price, get_employee_by_invoice_and_customer]

invoice_subagent_prompt = """
    You are a subagent among a team of assistants. You are specialized for retrieving and processing invoice information. You are routed for invoice-related portion of the questions, so only respond to them.. 

    You have access to three tools. These tools enable you to retrieve and process invoice information from the database. Here are the tools:
    - get_invoices_by_customer_sorted_by_date: This tool retrieves all invoices for a customer, sorted by invoice date.
    - get_invoices_sorted_by_unit_price: This tool retrieves all invoices for a customer, sorted by unit price.
    - get_employee_by_invoice_and_customer: This tool retrieves the employee information associated with an invoice and a customer.
    
    If you are unable to retrieve the invoice information, inform the customer you are unable to retrieve the information, and ask if they would like to search for something else.
    
    CORE RESPONSIBILITIES:
    - Retrieve and process invoice information from the database
    - Provide detailed information about invoices, including customer details, invoice dates, total amounts, employees associated with the invoice, etc. when the customer asks for it.
    - Always maintain a professional, friendly, and patient demeanor
    
    You may have additional context that you should use to help answer the customer's query. It will be provided to you below:
    """

invoice_information_subagent = create_react_agent(
    llm,
    tools=invoice_tools,
    name="invoice_information_subagent",
    prompt=invoice_subagent_prompt,
    state_schema=State,
    checkpointer=checkpointer,
    store=in_memory_store
)

# Supervisor setup
supervisor_prompt = """You are an expert customer support assistant for a digital music store. 
You are dedicated to providing exceptional service and ensuring customer queries are answered thoroughly. 
You have a team of subagents that you can use to help answer queries from customers. 
Your primary role is to serve as a supervisor/planner for this multi-agent team that helps answer queries from customers. 

Your team is composed of two subagents that you can use to help answer the customer's request:
1. music_catalog_information_subagent: this subagent has access to user's saved music preferences. It can also retrieve information about the digital music store's music 
catalog (albums, tracks, songs, etc.) from the database. 
3. invoice_information_subagent: this subagent is able to retrieve information about a customer's past purchases or invoices 
from the database. 

Based on the existing steps that have been taken in the messages, your role is to generate the next subagent that needs to be called. 
This could be one step in an inquiry that needs multiple sub-agent calls. """

# Import supervisor creation function
try:
    from langgraph_supervisor import create_supervisor
    
    supervisor_prebuilt_workflow = create_supervisor(
        agents=[invoice_information_subagent, music_catalog_subagent],
        output_mode="last_message",
        model=llm,
        prompt=supervisor_prompt,
        state_schema=State
    )
    
    supervisor_prebuilt = supervisor_prebuilt_workflow.compile(
        name="music_catalog_subagent", 
        checkpointer=checkpointer, 
        store=in_memory_store
    )
except ImportError:
    print("langgraph_supervisor not available, using basic implementation")
    supervisor_prebuilt = music_catalog_subagent

# Customer verification setup
class UserInput(BaseModel):
    """Schema for parsing user-provided account information."""
    identifier: str = Field(description="Identifier, which can be a customer ID, email, or phone number.")

structured_llm = llm.with_structured_output(schema=UserInput)

structured_system_prompt = """You are a customer service representative responsible for extracting customer identifier.\n 
Only extract the customer's account information from the message history. 
If they haven't provided the information yet, return an empty string for the file"""

def get_customer_id_from_identifier(identifier: str) -> Optional[int]:
    """Retrieve Customer ID using an identifier, which can be a customer ID, email, or phone number."""
    if identifier.isdigit():
        return int(identifier)
    
    elif identifier.startswith("+"):
        query = f"SELECT CustomerId FROM Customer WHERE Phone = '{identifier}';"
        result = db.run(query)
        formatted_result = ast.literal_eval(result)
        if formatted_result:
            return formatted_result[0][0]
    
    elif "@" in identifier:
        query = f"SELECT CustomerId FROM Customer WHERE Email = '{identifier}';"
        result = db.run(query)
        formatted_result = ast.literal_eval(result)
        if formatted_result:
            return formatted_result[0][0]
    
    return None 

def verify_info(state: State, config: RunnableConfig):
    """Verify the customer's account by parsing their input and matching it with the database."""
    if state.get("customer_id") is None: 
        system_instructions = """You are a music store agent, where you are trying to verify the customer identity 
        as the first step of the customer support process. 
        Only after their account is verified, you would be able to support them on resolving the issue. 
        In order to verify their identity, one of their customer ID, email, or phone number needs to be provided.
        If the customer has not provided the information yet, please ask them for it.
        If they have provided the identifier but cannot be found, please ask them to revise it."""

        user_input = state["messages"][-1] 
        parsed_info = structured_llm.invoke([SystemMessage(content=structured_system_prompt)] + [user_input])
        identifier = parsed_info.identifier
        
        customer_id = ""
        if identifier:
            customer_id = get_customer_id_from_identifier(identifier)
        
        if customer_id != "":
            intent_message = SystemMessage(
                content=f"Thank you for providing your information! I was able to verify your account with customer id {customer_id}."
            )
            return {
                "customer_id": customer_id,
                "messages": [intent_message]
            }
        else:
            response = llm.invoke([SystemMessage(content=system_instructions)] + state['messages'])
            return {"messages": [response]}
    else: 
        pass

def human_input(state: State, config: RunnableConfig):
    """ No-op node that should be interrupted on """
    user_input = interrupt("Please provide input.")
    return {"messages": [user_input]}

def should_interrupt(state: State, config: RunnableConfig):
    if state.get("customer_id") is not None:
        return "continue"
    else:
        return "interrupt"

# Memory management
def format_user_memory(user_data):
    """Formats music preferences from users, if available."""
    profile = user_data['memory']
    result = ""
    
    if hasattr(profile, 'music_preferences') and profile.music_preferences:
        result += f"Music Preferences: {', '.join(profile.music_preferences)}"
    
    return result.strip()

def load_memory(state: State, config: RunnableConfig, store: BaseStore):
    """Loads music preferences from users, if available."""
    user_id = state["customer_id"]
    namespace = ("memory_profile", user_id)
    
    existing_memory = store.get(namespace, "user_memory")
    
    formatted_memory = ""
    
    if existing_memory and existing_memory.value:
        formatted_memory = format_user_memory(existing_memory.value)

    return {"loaded_memory": formatted_memory}

class UserProfile(BaseModel):
    customer_id: str = Field(description="The customer ID of the customer")
    music_preferences: List[str] = Field(description="The music preferences of the customer")

create_memory_prompt = """You are an expert analyst that is observing a conversation that has taken place between a customer and a customer support assistant. The customer support assistant works for a digital music store, and has utilized a multi-agent team to answer the customer's request. 
You are tasked with analyzing the conversation that has taken place between the customer and the customer support assistant, and updating the memory profile associated with the customer. The memory profile may be empty. If it's empty, you should create a new memory profile for the customer.

You specifically care about saving any music interest the customer has shared about themselves, particularly their music preferences to their memory profile.

To help you with this task, I have attached the conversation that has taken place between the customer and the customer support assistant below, as well as the existing memory profile associated with the customer that you should either update or create. 

The customer's memory profile should have the following fields:
- customer_id: the customer ID of the customer
- music_preferences: the music preferences of the customer

These are the fields you should keep track of and update in the memory profile. If there has been no new information shared by the customer, you should not update the memory profile. It is completely okay if you do not have new information to update the memory profile with. In that case, just leave the values as they are.

*IMPORTANT INFORMATION BELOW*

The conversation between the customer and the customer support assistant that you should analyze is as follows:
{conversation}

The existing memory profile associated with the customer that you should either update or create based on the conversation is as follows:
{memory_profile}

Ensure your response is an object that has the following fields:
- customer_id: the customer ID of the customer
- music_preferences: the music preferences of the customer

For each key in the object, if there is no new information, do not update the value, just keep the value that is already there. If there is new information, update the value. 

Take a deep breath and think carefully before responding.
"""

def create_memory(state: State, config: RunnableConfig, store: BaseStore):
    user_id = str(state["customer_id"])
    namespace = ("memory_profile", user_id)
    
    existing_memory = store.get(namespace, "user_memory")
    
    formatted_memory = ""
    if existing_memory and existing_memory.value:
        existing_memory_dict = existing_memory.value
        formatted_memory = (
            f"Music Preferences: {', '.join(existing_memory_dict.get('memory').music_preferences or [])}"
        )

    formatted_system_message = SystemMessage(content=create_memory_prompt.format(
        conversation=state["messages"], 
        memory_profile=formatted_memory
    ))
    
    updated_memory = llm.with_structured_output(UserProfile).invoke([formatted_system_message])
    
    key = "user_memory"
    store.put(namespace, key, {"memory": updated_memory})

# Build the final graph
multi_agent_final = StateGraph(State)

multi_agent_final.add_node("verify_info", verify_info)
multi_agent_final.add_node("human_input", human_input)
multi_agent_final.add_node("load_memory", load_memory)
multi_agent_final.add_node("supervisor", supervisor_prebuilt)
multi_agent_final.add_node("create_memory", create_memory)

multi_agent_final.add_edge(START, "verify_info")
multi_agent_final.add_conditional_edges(
    "verify_info",
    should_interrupt,
    {
        "continue": "load_memory",
        "interrupt": "human_input",
    },
)
multi_agent_final.add_edge("human_input", "verify_info")
multi_agent_final.add_edge("load_memory", "supervisor")
multi_agent_final.add_edge("supervisor", "create_memory")
multi_agent_final.add_edge("create_memory", END)

# Export the final graph
graph = multi_agent_final.compile(
    name="multi_agent_final", 
    checkpointer=checkpointer, 
    store=in_memory_store
)