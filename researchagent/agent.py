# %% [markdown]
# ### Capstone project
# ### Create agent framework to implement rule engine to build a stock portfolio by reasearching stocks based on fundamental and technical criterias.
# ### Create fundamental anaylsis agents to analyze balance sheet, income statement, cash flow statement for stock, 
# ### aggregator agent to calculate all ratios, agent to calculate valuation using DCF method. 
# ### create technical analysis agents to provide technical analysis ratios
# ### user should be able to say: select stocks with PEG > 1.0 and correlation >0.7
# ### See the stocks that are matching. 
# ### Then add to criteria saying: add filter to select stock with ROA > 20% and P/E < 15
# ### Later expand to connect to a broker and 
# ### 1. get current portfolio positions
# ### 2. submit buy/sell orders on behalf of customer

# %%
# Standard library imports
import json
import requests
import subprocess
import time
import uuid
import warnings
from typing import Any, Dict

# Third-party imports
from mcp import StdioServerParameters

# Google ADK imports
from google.adk.a2a.utils.agent_to_a2a import to_a2a
from google.adk.agents import (
    Agent, 
    LlmAgent, 
    LoopAgent, 
    ParallelAgent, 
    SequentialAgent
)
from google.adk.agents.remote_a2a_agent import (
    RemoteA2aAgent,
    AGENT_CARD_WELL_KNOWN_PATH,
)
from google.adk.apps.app import App, EventsCompactionConfig, ResumabilityConfig
from google.adk.code_executors import BuiltInCodeExecutor
from google.adk.memory import InMemoryMemoryService
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner, Runner
from google.adk.sessions import DatabaseSessionService, InMemorySessionService
from google.adk.tools import (
    AgentTool, 
    FunctionTool, 
    google_search, 
    load_memory, 
    preload_memory,
    ToolContext
)
from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from google.adk.tools.tool_context import ToolContext
from google.genai import types
from google.adk.agents.invocation_context import InvocationContext

# Hide additional warnings in the notebook
warnings.filterwarnings("ignore")

print("✅ ADK components imported successfully.")

# %%
import os

# Detect environment and configure API key accordingly
def setup_google_api_key():
    try:
        # Try Kaggle environment first
        from kaggle_secrets import UserSecretsClient
        GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
        os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
        print("✅ Gemini API key loaded from Kaggle Secrets.")
        return GOOGLE_API_KEY
    except ImportError:
        # Not in Kaggle, try environment variable
        GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
        if GOOGLE_API_KEY:
            print("✅ Gemini API key loaded from environment variable.")
            return GOOGLE_API_KEY
        else:
            raise ValueError(
                "🔑 GOOGLE_API_KEY not found. Please either:\n"
                "- Set GOOGLE_API_KEY environment variable (local), or\n"
                "- Add GOOGLE_API_KEY to Kaggle Secrets (Kaggle environment)"
            )
    except Exception as e:
        # Kaggle import worked but secret retrieval failed
        print(f"⚠️  Kaggle Secrets error: {e}")
        # Fall back to environment variable
        GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
        if GOOGLE_API_KEY:
            print("✅ Gemini API key loaded from environment variable (fallback).")
            return GOOGLE_API_KEY
        else:
            raise ValueError(
                "🔑 GOOGLE_API_KEY not available. Please either:\n"
                "- Add 'GOOGLE_API_KEY' to your Kaggle secrets, or\n"
                "- Set GOOGLE_API_KEY environment variable"
            )

# Setup the API key
GOOGLE_API_KEY = setup_google_api_key()

# %% [markdown]
# ### 1.4: Helper functions
# 
# We'll define some helper functions. If you are running this outside the Kaggle environment, you don't need to do this.

# %%
# Define helper functions that will be reused throughout the notebook
from IPython.display import display, HTML
try:
    from jupyter_server.serverapp import list_running_servers
except ImportError:
    # Fallback for older Jupyter environments
    try:
        from notebook.notebookapp import list_running_servers
    except ImportError:
        # Alternative for environments without notebook server access
        def list_running_servers():
            return [{"base_url": "/", "port": 8888, "token": ""}]

# Gets the proxy URL for local Jupyter environment
def get_adk_proxy_url():
    ADK_PORT = "8000"
    
    try:
        servers = list(list_running_servers())
        if servers:
            server = servers[0]
            base_url = server.get("base_url", "/")
            port = server.get("port", 8888)
            token = server.get("token", "")
            
            # For local Jupyter, construct simple localhost URL
            if token:
                url = f"http://localhost:{ADK_PORT}/?token={token}"
            else:
                url = f"http://localhost:{ADK_PORT}"
        else:
            # Default fallback for local development
            url = f"http://localhost:{ADK_PORT}"
    except Exception:
        # Fallback URL for local development
        url = f"http://localhost:{ADK_PORT}"

    styled_html = f"""
    <div style="padding: 15px; border: 2px solid #f0ad4e; border-radius: 8px; background-color: #fef9f0; margin: 20px 0;">
        <div style="font-family: sans-serif; margin-bottom: 12px; color: #333; font-size: 1.1em;">
            <strong>⚠️ IMPORTANT: Action Required</strong>
        </div>
        <div style="font-family: sans-serif; margin-bottom: 15px; color: #333; line-height: 1.5;">
            The ADK web UI is <strong>not running yet</strong>. You must start it in the next cell.
            <ol style="margin-top: 10px; padding-left: 20px;">
                <li style="margin-bottom: 5px;"><strong>Run the next cell</strong> (the one with <code>!adk web ...</code>) to start the ADK web UI.</li>
                <li style="margin-bottom: 5px;">Wait for that cell to show it is "Running" (it will not "complete").</li>
                <li>Once it's running, <strong>return to this button</strong> and click it to open the UI.</li>
            </ol>
            <em style="font-size: 0.9em; color: #555;">(If you click the button before running the next cell, you will get a 500 error.)</em>
        </div>
        <a href='{url}' target='_blank' style="
            display: inline-block; background-color: #1a73e8; color: white; padding: 10px 20px;
            text-decoration: none; border-radius: 25px; font-family: sans-serif; font-weight: 500;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2); transition: all 0.2s ease;">
            Open ADK Web UI (after running cell below) ↗
        </a>
    </div>
    """

    display(HTML(styled_html))

    return ""  # Return empty string since local ADK doesn't need url_prefix


print("✅ Helper functions defined.")

# %% [markdown]
# ### 1.5: Configure Retry Options
# 
# When working with LLMs, you may encounter transient errors like rate limits or temporary service unavailability. Retry options automatically handle these failures by retrying the request with exponential backoff.

# %%
retry_config=types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1, # Initial delay before first retry (in seconds)
    http_status_codes=[429, 500, 503, 504] # Retry on these HTTP errors
)

# %%
import logging
import os

# Clean up any previous logs
for log_file in ["logger.log", "web.log", "tunnel.log"]:
    if os.path.exists(log_file):
        os.remove(log_file)
        print(f"🧹 Cleaned up {log_file}")

# Configure logging with DEBUG log level.
logging.basicConfig(
    filename="logger.log",
    level=logging.DEBUG,
    format="%(filename)s:%(lineno)s %(levelname)s:%(message)s",
)

# %%
# Define helper functions that will be reused throughout the notebook
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


print("✅ Helper functions defined.")

# %%
def get_available_stocks() -> list:
    """Gives the master list of available stocks.

    This tool simulates querying database to retrieve list of available stocks.

    Args:
        none

    Returns:
        List with stock tickers .
        Success: {"AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", etc}
        Error: {""}
    """
    # This simulates getting the master list of available stocks.
    stock_list = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "UPS", "INTC", "CMG", "WEN", "F", "GAP", "SBUX"]

    return stock_list

print("✅ get_available_stocks function created")
print(f"💳 Test: {get_available_stocks()}")

# %%
def get_stock_technical_data(ticker: str) -> dict:
    """Gives the technical analysis data for stocks and technical analysis ratios like correlation, 
     13 day ema, 50 day ema, macd fast, macd slow, trend, rsi and peg for a given stock ticker.

    This tool simulates getting a realtime data for a stock and pre-calculated technical ratios.
    Args:
        ticker: The stock ticker symbol, e.g., "AAPL" for Apple Inc.

    Returns:
        Dict with realtime stock data .
        Success: {"ticker": "AAPL", "price": 150.25, "volume": 1000000, "correlation": 0.85, ema_13: 145.0, ema_50: 148.0,
            "macd_fast": 1.5, "macd_slow": 1.2, "trend": "upward", "rsi": 70}
        Error: {""}
    """
    # This simulates getting the master list of available stocks.
    rtdata =[
        {"ticker": "AAPL", "price": 150.25, "volume": 1000000, "correlation": 0.85, "ema_13": 145.0, "ema_50": 148.0,
            "macd_fast": 1.5, "macd_slow": 1.2, "trend": "upward", "rsi": 70, "peg": 1.8},
        {"ticker": "GOOGL", "price": 2800.50, "volume": 800000, "correlation": 0.75, "ema_13": 2750.0, "ema_50": 2780.0,
            "macd_fast": 2.0, "macd_slow": 1.8, "trend": "upward", "rsi": 65, "peg": 1.2},
        {"ticker": "MSFT", "price": 300.10, "volume": 1200000, "correlation": 0.80, "ema_13": 295.0, "ema_50": 298.0,
            "macd_fast": 1.0, "macd_slow": 0.8, "trend": "upward", "rsi": 60, "peg": 2.0},
        {"ticker": "AMZN", "price": 3500.75, "volume": 600000, "correlation": 0.70, "ema_13": 3450.0, "ema_50": 3480.0,
            "macd_fast": 2.5, "macd_slow": 2.2, "trend": "upward", "rsi": 55, "peg": 2.5},
        {"ticker": "TSLA", "price": 700.30, "volume": 900000, "correlation": 0.90, "ema_13": 690.0, "ema_50": 695.0,
            "macd_fast": 3.0, "macd_slow": 2.7, "trend": "upward", "rsi": 75, "peg": 3.8},
        {"ticker": "UPS", "price": 180.50, "volume": 500000, "correlation": 0.65, "ema_13": 175.0, "ema_50": 178.0,
        "macd_fast": 0.5, "macd_slow": 0.3, "trend": "upward", "rsi": 50,   "peg": 1.1},
        {"ticker": "INTC", "price": 55.25, "volume": 1100000, "correlation": 0.60, "ema_13": 54.0, "ema_50": 54.5,
        "macd_fast": 0.2, "macd_slow": 0.1, "trend": "upward", "rsi": 45, "peg": 1.3},
        {"ticker": "CMG", "price": 1500.75, "volume": 400000, "correlation": 0.55, "ema_13": 1480.0, "ema_50": 1490.0,
        "macd_fast": 1.8, "macd_slow": 1.5, "trend": "upward", "rsi": 68, "peg": 0.95},
        {"ticker": "WEN", "price": 25.30, "volume": 1300000, "correlation": 0.50, "ema_13": 24.5, "ema_50": 24.8,
        "macd_fast": 0.1, "macd_slow": 0.05, "trend": "upward", "rsi": 40, "peg": 1.4},
        {"ticker": "F", "price": 15.10, "volume": 1400000, "correlation": 0.45, "ema_13": 14.8, "ema_50": 14.9,
        "macd_fast": 0.05, "macd_slow": 0.02, "trend": "upward", "rsi": 35,     "peg": 1.6},
        {"ticker": "GAP", "price": 30.75, "volume": 700000, "correlation": 0.40, "ema_13": 30.0, "ema_50": 30.5,
        "macd_fast": 0.15, "macd_slow": 0.1, "trend": "upward", "rsi": 55, "peg": 1.2},
        {"ticker": "SBUX", "price": 120.50, "volume": 950000, "correlation": 0.35, "ema_13": 118.0, "ema_50": 119.0,
        "macd_fast": 0.8, "macd_slow": 0.6, "trend": "upward", "rsi": 62, "peg": 0.85}
    ]

    #find the data for the requested ticker
    stockdata={}
    for stock in rtdata:
        if stock["ticker"] == ticker:
            stockdata = stock    

    return stockdata

print("✅ get_stock_technical_data function created")
print(f"💳 Test: {get_stock_technical_data('AAPL')}")

# %%
parser_agent = LlmAgent(
    name="PromptParser",
    instruction="Analyze the user's prompt and extract ONLY the stock ticker as a single word. Store it in state['ticker']. Do not respond to the user.",
    output_key="ticker", # Store the output directly in state['ticker']
    model="gemini-2.5-flash-lite"
)




# %%
def balancesheet_focused_instruction(ctx: InvocationContext):
    ticker = ctx.state.get("ticker", "?")
    print("✅ Retrieved ticker in BalanceSheetAgent:", ticker)
    return f'''Get Balance Sheet data for {ticker} and organize in a python dictionary key value pair for further processing in python. 
    Your only output strictly should be a dictionary with key value pairs with balance sheet fields and nothing else.'''

balanceSheetAgent = Agent(
    name="BalanceSheetAnalyst",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        api_key=GOOGLE_API_KEY,
        retry_options=retry_config
    ),
    description="An agent specialized in analyzing balance sheets.",
    instruction=balancesheet_focused_instruction,
    tools=[google_search],
    output_key="balanceSheet"
)

print("✅ BalanceSheetAnalyst Agent defined.")

# %%
def incomestatement_focused_instruction(ctx: InvocationContext):
    ticker = ctx.state.get("ticker", "?")
    print("✅ Retrieved ticker in IncomeStatementAgent:", ticker)
    return f'''Get Income Statement data for {ticker} and organize in a python dictionary key value pair for further processing in python. 
    Your only output strictly should be the dictionary with key value pairs with incomestatement fields and nothing else.'''

incomeStatementAgent = Agent(
    name="IncomeStatementAnalyst",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        api_key=GOOGLE_API_KEY,
        retry_options=retry_config
    ),
    description="An agent specialized in analyzing income statements.",
    instruction=incomestatement_focused_instruction,
    tools=[google_search],
    output_key="incomeStatement"
)

print("✅ IncomeStatementAnalyst Agent defined.")

# %%
def cashflowstatement_focused_instruction(ctx: InvocationContext):
    ticker = ctx.state.get("ticker", "?")
    print("✅ Retrieved ticker in CashFlowAgent:", ticker)
    return f'''Get Cash Flow Statement data for {ticker} and organize in a python dictionary key value pair for further processing in python. 
    Your only output strictly should be the dictionary with key value pairs with cashflow fields.'''

cashFlowAgent = Agent(
    name="CashFlowAnalyst",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        api_key=GOOGLE_API_KEY,
        retry_options=retry_config
    ),
    description="An agent specialized in analyzing cash flow statements.",
    instruction=cashflowstatement_focused_instruction,
    tools=[google_search],
    output_key="cashFlowStatement"
)

print("✅ CashFlowAnalyst Agent defined.")

# %%
dcfAgent = Agent(
    name="DCFAnalyst",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        api_key=GOOGLE_API_KEY,
        retry_options=retry_config
    ),
    description="An agent specialized in analyzing discounted cash flow (DCF).",
    instruction='''Calculate the valuation using discounted cash flow (DCF) method. Get any required data needed.
    Return only the DCF valuation as a single numeric value. Do not give any other summary text or ratios from other agents.''',
    tools=[google_search],
    output_key="DCFvaluation"
)

print("✅ CashFlowAnalyst Agent defined.")

# %%
# The AggregatorAgent runs *after* the parallel step to synthesize the results.
aggregator_agent = Agent(
    name="aggregator_agent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    description='''An agent specialized in calculating profitability and valuation ratios using Balance sheet, Income statement and Cash flow statements.
    ''',
    # It uses placeholders to inject the outputs from the parallel agents, which are now in the session state.
    instruction="""
    If stock ticker is not given, use available stocks tool.
    Get balance sheet data using balanceSheetAgent. 
    Get income statement data using incomeStatementAgent. 
    Get cash flow statement data using cashFlowAgent. 
    Calculate the below profitability ratios for given stock ticker by using the data retrieved.
        a. Gross Profit Margin: (Revenue - Cost of Goods Sold) / Revenue.
        b. Operating Profit Margin: Operating Profit / Revenue.
        c. Net Profit Margin: Net Income / Revenue.
        d. Return on Assets (ROA): Net Income / Total Assets.
    Calculate the below valuation ratios for given stock ticker by using the data retrieved.
        a. Price to Earnings (P/E) Ratio: Market Value per Share / Earnings per Share (EPS).
        b. Price to Book (P/B) Ratio: Market Value per Share / Book Value per Share.
        c. Price to Sales (P/S) Ratio: Market Value per Share / Revenue per Share.
        d. Enterprise Value to EBITDA (EV/EBITDA): Enterprise Value / Earnings Before Interest, Taxes, Depreciation, and Amortization.
    Give only the aggregated profitability and valuation ratios as key value pairs. 
    Do not give any other summary text or ratios from other agents.        
    """,
     tools=[
        AgentTool(agent=balanceSheetAgent),
        AgentTool(agent=incomeStatementAgent),
        AgentTool(agent=cashFlowAgent),
        get_available_stocks
    ]
)

print("✅ aggregator_agent created.")

# %%
# This SequentialAgent defines the high-level workflow: run the parallel team first, then run the aggregator.
from vertexai.agent_engines import AdkApp
def session_service_builder():
    return InMemorySessionService()

root_agent = LlmAgent(
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    name="StockResearchSystem",
    instruction='''Prompt user for portfolio selection criteria to start with. 
    Build stock portfolio criteria interactively by asking user for selection criteria.
    If stock ticker is not given, use available stocks tool.
    Get required technical analysis data for stocks and fundamental data like balance sheet, income statement, cash flow statement. 
    Calculate profitability and valuation ratios.
    Use calculated ratios and execute any equations to filter stocks. 
    Ask user for refining criteria until user is satisfied.
    Always use the latest selected stocks matching criteria to refine further.
    Provide the list of stocks matching the criteria.
    Do not provide any other summary text.
    Use load_memory tool if you need to recall past conversations.''',
    tools=[
        AgentTool(agent=parser_agent),
        AgentTool(agent=balanceSheetAgent),
        AgentTool(agent=incomeStatementAgent),
        AgentTool(agent=cashFlowAgent),
        AgentTool(agent=aggregator_agent),
        get_available_stocks,
        get_stock_technical_data,
        load_memory
    ],  
)

app = AdkApp(
    agent=root_agent,
    session_service_builder=session_service_builder,
)

