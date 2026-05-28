import asyncio
import json
import os
import subprocess
from datetime import datetime, timedelta, timezone
from typing import Annotated, Any, Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
import signal

# 一樣建立agent結構
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


# mcp設定，並且開啟calendar權限
WORKSPACE_MCP_CONFIG = {
    "workspace": {
        "command": "python",
        "args": [
            os.path.expanduser(
                "~/lion_stuff/LLM_class/project/Workspace_Agent/google_workspace_mcp/main.py"
            ),
            "--tool-tier", "extended",
            "--permissions", "calendar:full",  # Calendar 完整權限
        ],
        "transport": "stdio",
        "env": {
            "GOOGLE_OAUTH_CLIENT_ID": os.getenv("GOOGLE_OAUTH_CLIENT_ID", ""),
            "GOOGLE_OAUTH_CLIENT_SECRET": os.getenv("GOOGLE_OAUTH_CLIENT_SECRET", ""),
            "OAUTHLIB_INSECURE_TRANSPORT": "1",
            "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", ""),
            "USER_GOOGLE_EMAIL": os.getenv("USER_GOOGLE_EMAIL", ""),
            "CREDENTIALS_DIR": os.path.expanduser("~/.google_workspace_mcp/credentials"),
        },
    }
}

# 定義工具
ALLOWED_CALENDAR_TOOLS = {
    "list_calendars",       # 列出所有行事曆
    "get_events",           # 列出/查詢事件
    "manage_event",         # 建立、修改、刪除事件
    "query_freebusy",       # 查詢空閒時段（PDF 的 suggest_meeting_time）
    "create_calendar",      # 建立新行事曆
}

# 跑CLI用的

# 取得OAUTH授權檔案
def _get_calendar_service():
    
    user_email = os.getenv("USER_GOOGLE_EMAIL", "")
    credentials_dir = os.path.expanduser(
        os.getenv("CREDENTIALS_DIR", "~/.google_workspace_mcp/credentials")
    )
    token_path = os.path.join(credentials_dir, f"{user_email}.json")
    with open(token_path) as f:
        token_data = json.load(f)
    
    creds = Credentials(
        token=token_data["token"],
        refresh_token=token_data["refresh_token"],
        token_uri=token_data["token_uri"],
        client_id=token_data["client_id"],
        client_secret=token_data["client_secret"],
        scopes=token_data["scopes"],
    )
    return build("calendar", "v3", credentials=creds)



# CLI @tool


# 取得行事曆行程
@tool
def cli_today_events(calendar_id: str = "primary") -> dict:
    """Get today's calendar events using Google Calendar API directly."""
    def timeout_handler(signum, frame):
        raise TimeoutError("CLI tool timed out after 15 seconds")
    try:
        
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(15)
        
        service = _get_calendar_service()
        now = datetime.now(timezone.utc)
        time_min = now.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
        time_max = now.replace(hour=23, minute=59, second=59, microsecond=0).isoformat()
        
        result = service.events().list(
            calendarId=calendar_id,
            timeMin=time_min,
            timeMax=time_max,
            singleEvents=True,
            orderBy="startTime"
        ).execute()
        signal.alarm(0)
        return result
    except Exception as e:
        signal.alarm(0)
        return {"error": str(e)}


# 列出指定範圍內的行事曆事件
@tool
def cli_list_events(
    time_min: str = "",
    time_max: str = "",
    max_results: int = 10,
    calendar_id: str = "primary"
) -> dict:
    """List calendar events within a date range using Google Calendar API directly."""
    def timeout_handler(signum, frame):
        raise TimeoutError("CLI tool timed out after 15 seconds")
    try:
        
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(15)

        service = _get_calendar_service()
        if not time_min:
            time_min = datetime.now(timezone.utc).isoformat()
        if not time_max:
            time_max = (datetime.now(timezone.utc) + timedelta(days=7)).isoformat()
        
        result = service.events().list(
            calendarId=calendar_id,
            timeMin=time_min,
            timeMax=time_max,
            maxResults=max_results,
            singleEvents=True,
            orderBy="startTime"
        ).execute()
        signal.alarm(0)
        return result
    except Exception as e:
        signal.alarm(0)
        return {"error": str(e)}

# 列出所有行事曆

@tool
def cli_list_calendars() -> dict:
    """List all calendars using Google Calendar API directly."""
    def timeout_handler(signum, frame):
        raise TimeoutError("CLI tool timed out after 15 seconds")
    try:
        
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(15)
        service = _get_calendar_service()
        result = service.calendarList().list().execute()
        signal.alarm(0)
        return result

    except Exception as e:
        signal.alarm(0)
        return {"error": str(e)}

# 取得單一行事曆的詳細資料
@tool
def cli_get_event(event_id: str, calendar_id: str = "primary") -> dict:
    """Get details of a single calendar event using Google Calendar API directly."""
    def timeout_handler(signum, frame):
        raise TimeoutError("CLI tool timed out after 15 seconds")
    try:
        
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(15)
        service = _get_calendar_service()
        result = service.events().get(
            calendarId=calendar_id,
            eventId=event_id
        ).execute()
        signal.alarm(0)
        return result
    except Exception as e:
        signal.alarm(0)
        return {"error": str(e)}


# 列出目前有的cli工具
@tool
def cli_tool_list() -> dict:
    """List all available CLI tools for debugging purposes."""
    return {
        "tools": [
            "cli_today_events",
            "cli_list_events", 
            "cli_list_calendars",
            "cli_get_event",
            "cli_tool_list"
        ]
    }

# Prompt設定
def get_system_prompt() -> SystemMessage:
    """
    動態產生 System Prompt，插入今天的日期
    讓 Agent 在回答「今天」、「這週」等問題時有正確的時間基準
    """
    today = datetime.now().strftime("%A, %B %d, %Y")
    timezone_str = "Asia/Taipei"
    
    return SystemMessage(content=f"""
You are an intelligent calendar assistant that helps users manage their Google Calendar.
Today's date is {today}.
User's timezone is {timezone_str}. Always include timezone in all calendar event times when calling tools.
When creating or updating events, always set:
- start.timeZone: "{timezone_str}"
- end.timeZone: "{timezone_str}"

## Tool Selection Guide

### Use CLI tools for FAST READ-ONLY queries:
- cli_today_events: "What's on today?", "Do I have anything today?"
- cli_list_events: "Show me this week", "What's on next Monday?"
- cli_list_calendars: "What calendars do I have?", "List my calendars"
- cli_get_event: "Get details for that meeting", "Tell me more about that event"
- cli_tool_list: Debug purposes only

### Use MCP tools for FULL CRUD operations:
- get_events: Complex event queries with filtering
- manage_event: "Schedule a meeting", "Add an event", "Move the meeting", "Delete the meeting"
- query_freebusy: "Find a free slot", "When are we both available?"
- list_calendars: List all calendars
- create_calendar: Create a new calendar

## Output Formatting Rules
- Always convert ISO timestamps to human-readable format (e.g., "Monday, June 3 at 2:00 PM")
- Show event duration when relevant
- List events in chronological order
- For empty results, clearly state "No events found"

## Safety Rules
- ALWAYS ask for confirmation before delete_calendar_event or update_calendar_event
- Example: "Are you sure you want to delete 'Team Standup' on Monday?"
- Only proceed with destructive operations after explicit user confirmation
- ALWAYS call get_events first to retrieve the real event ID before calling manage_event to delete or update
- NEVER use placeholder IDs like 'team-meeting-id', always use the actual event ID from get_events results

## Response Format
- Be concise and friendly
- Use bullet points for listing multiple events
- Include location and attendees when relevant
- Suggest follow-up actions when appropriate
""")



# 建立Agent
# 會有兩種工具可以使用
# - MCP tools：從server動態載入，用於建立、讀取、更新、刪除操作
# - CLI tools：本地@tool函數，用於快速讀取

async def build_agent(mcp_client: MultiServerMCPClient):
    
    
    # 從MCP載入calendar工具
    all_mcp_tools = await mcp_client.get_tools()
    
    # 保留需要的工具
    calendar_mcp_tools = [
        t for t in all_mcp_tools
        if t.name in ALLOWED_CALENDAR_TOOLS
    ]
    print(f"[Agent] Loaded {len(calendar_mcp_tools)} Calendar MCP tools: "
          f"{[t.name for t in calendar_mcp_tools]}")
    
    # 定義 CLI @tool
    cli_tools = [
        cli_today_events,
        cli_list_events,
        cli_list_calendars,
        cli_get_event,
        cli_tool_list,
    ]
    print(f"[Agent] Loaded {len(cli_tools)} CLI tools: "
          f"{[t.name for t in cli_tools]}")
    
    # 合併兩個工具層讓模型選擇
    all_tools = calendar_mcp_tools + cli_tools
    print(f"[Agent] Total tools available: {len(all_tools)}")
    
    # 建立 GPT-4o 
    llm = ChatOpenAI(model="gpt-4o", temperature=0).bind_tools(all_tools)
    
    # 定義節點
    
    # agent節點
    def agent_node(state: AgentState) -> AgentState:
        system_prompt = get_system_prompt()
        print(f"[Thinking] GPT-4o reasoning... (messages in state: {len(state['messages'])})")
        response = llm.invoke([system_prompt] + list(state["messages"]))
    
        # debug用的
        if response.tool_calls:
            for tc in response.tool_calls:
                print(f"[Tool Call] → {tc['name']}({tc['args']})")
        else:
            print("[Final] GPT-4o returning final answer")
    
        return {"messages": [response]}
    
    # 條件節點
    def should_continue(state: AgentState) -> str:
        
        last_message = state["messages"][-1]
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "tools"
        return END
    
    # 建圖 
    
    graph = StateGraph(AgentState)
    
    # 加入兩個節點
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(all_tools))  # ToolNode 處理兩個工具層
    
    # 設定入口和邊
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue)
    graph.add_edge("tools", "agent")  # 工具執行完永遠回到 agent
    
    return graph.compile()


# Demo用的會有三個模式測試
# 1. cli_list_calendars
# 2. cli_today_events
# 3. cli_list_events

async def run_demo(agent) -> None:
    
    demo_queries = [
        "What calendars do I have?",
        "What's on my calendar today?",
        "Show me my events for the next 7 days.",
    ]
    
    print("\n" + "="*60)
    print("Running Demo Mode...")
    print("="*60)
    
    for i, query in enumerate(demo_queries, 1):
        print(f"\n[Demo {i}/3] {query}")
        print("-" * 40)
        
        result = await agent.ainvoke({
            "messages": [HumanMessage(content=query)]
        })
        
        final_response = result["messages"][-1]
        print(f"Agent: {final_response.content}")


# 互動版，一樣會紀錄上下文
async def run_interactive_chat(agent) -> None:
    
    print("\n" + "="*60)
    print("Calendar Agent - Interactive Mode")
    print("Commands: 'demo' to run demo, 'exit' to quit")
    print("="*60 + "\n")
    
    # 對話歷史
    history = []
    
    while True:
        user_input = input("You: ").strip()
        
        if not user_input:
            continue
            
        # 有特別設定exit跟demo指令來執行
        if user_input.lower() in ("exit", "quit"):
            print("Session ended.")
            break
        if user_input.lower() == "demo":
            await run_demo(agent)
            continue
        
        # 把使用者輸入加入歷史
        history.append(HumanMessage(content=user_input))
        
        # 傳入完整歷史
        result = await agent.ainvoke({"messages": history})
        
        # 取出最後一則回應
        ai_response = result["messages"][-1]
        print(f"\nAgent: {ai_response.content}\n")
        
        # 把最後的回應也加入歷史
        history.append(ai_response)



# 環境變數

def validate_env() -> bool:
    
    required_vars = {
        "OPENAI_API_KEY": "OpenAI API 金鑰",
        "GOOGLE_OAUTH_CLIENT_ID": "Google OAuth Client ID",
        "GOOGLE_OAUTH_CLIENT_SECRET": "Google OAuth Client Secret",
    }
    
    missing = []
    for var, description in required_vars.items():
        if not os.getenv(var):
            missing.append(f"  - {var} ({description})")
    
    if missing:
        print("錯誤：缺少以下環境變數：")
        for m in missing:
            print(m)
        print("\n請在 .env 檔案中設定後重新執行。")
        return False
    
    return True



# 主程式

async def main() -> None:
    
    if not validate_env():
        return
    
    print("Starting Calendar Agent...")
    print("Connecting to Google Workspace MCP Server...")
    
    # 建立 MCP client
    mcp_client = MultiServerMCPClient(WORKSPACE_MCP_CONFIG)
    
    # 建立 Agent
    agent = await build_agent(mcp_client)
    print("Agent ready!\n")
    
    # 選擇執行模式
    print("Select mode:")
    print("  1. Interactive mode - chat with the agent")
    print("  2. Demo mode - run 3 preset queries")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == "1":
        await run_interactive_chat(agent)
    elif choice == "2":
        await run_demo(agent)
    else:
        print("Invalid choice. Running interactive mode by default.")
        await run_interactive_chat(agent)


if __name__ == "__main__":
    asyncio.run(main())
