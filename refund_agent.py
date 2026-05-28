import asyncio
import os
from typing import Annotated, Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


# 透過LangGraph來設定模型狀態
# 只需要存message，並且把後續的訊息加上去
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


# 設定MCP來讓客戶端可以存取模型
WORKSPACE_MCP_CONFIG = {
    "workspace": {
        "command": "python",
        "args": [
	    os.path.expanduser(# 先前設定的mcp伺服器資料夾
		"~/lion_stuff/LLM_class/project/Workspace_Agent/google_workspace_mcp/main.py"
	    ),
	    "--tool-tier", "core",
	    "--permissions", "gmail:send",
	],
        "transport": "stdio",# 標準輸入輸出，將資料留在本機
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

# 此專案會用到的gmail工具
ALLOWED_GMAIL_TOOLS = {
    "search_gmail_messages",          # 搜尋收件匣
    "get_gmail_message_content",      # 讀取單封信件完整內容
    "get_gmail_messages_content_batch",  # 批次讀取多封信件
    "send_gmail_message",             # 送出回信
    "create_gmail_draft",             # 建立草稿
    "get_gmail_thread",               # 讀取完整對話串
    "list_gmail_labels",              # 列出所有標籤
}

# 設定模型prompt
SYSTEM_PROMPT = SystemMessage(content="""
You are an autonomous customer service agent responsible for processing refund and return emails.

## Your 6-Step Workflow
Execute these steps in order for every run:
1. SEARCH - Search Gmail inbox for emails with query: "refund OR return OR complaint is:unread"
2. READ - Read the full content of each email found
3. CLASSIFY - Classify each email into one of four categories
4. DRAFT - Compose a reply using the appropriate template below
5. SEND - Send the reply as a threaded response using the original thread_id
6. REPORT - Print a summary of all emails processed

## Email Classification Rules
- REFUND_REQUEST: Customer explicitly asks for money back
- RETURN_REQUEST: Customer wants to return a product
- COMPLAINT: Customer expresses dissatisfaction but does not explicitly request refund/return
- OTHER: Unrelated content (promotions, spam, newsletters) — DO NOT reply to these

## Reply Templates

### REFUND_REQUEST Template:
Subject: Re: [original subject]
Body:
Dear [Customer Name],

Thank you for reaching out to us. We have received your refund request and are happy to assist.

Your refund has been approved and will be processed within 3-5 business days. 
The amount will be credited back to your original payment method.

If you have any questions, please don't hesitate to contact us.

Best regards,
Customer Service Team

### RETURN_REQUEST Template:
Subject: Re: [original subject]
Body:
Dear [Customer Name],

Thank you for contacting us regarding your return request.

Please follow these steps to return your item:
1. Pack the item securely in its original packaging if possible
2. A prepaid return label will be emailed to you within 24 hours
3. Drop off the package at any authorized shipping location
4. Your refund will be processed within 5-7 business days after we receive the item

Best regards,
Customer Service Team

### COMPLAINT Template:
Subject: Re: [original subject]
Body:
Dear [Customer Name],

Thank you for bringing this to our attention. We sincerely apologize for the inconvenience you experienced.

We take all customer feedback seriously and will investigate this matter immediately.
A member of our team will follow up with you within 24 hours with a resolution.

Best regards,
Customer Service Team

## Critical Rules
- ALWAYS use thread_id when sending replies to ensure proper email threading
- NEVER reply to emails classified as OTHER
- When uncertain about classification, use create_gmail_draft instead of send_gmail_message
- Process ALL unread emails matching the search query, not just the first one
- Include the customer's name in the reply if available from the email
- ALWAYS use the exact message_id returned from search_gmail_messages results
- NEVER use placeholder or invented message IDs
- If you need to read an email, first search for it to get the real message_id
""")

# 建立整個agent的架構
#    1. 從 MCP server 取得所有可用工具
#    2. 過濾只保留 Gmail 相關工具
#    3. 建立 GPT-4o 並綁定工具（native function calling）
#    4. 定義三個節點和條件路由
#    5. 編譯並回傳可執行的 graph

async def build_agent(mcp_client: MultiServerMCPClient):
    
    # 動態取得需要的工具清單
    all_mcp_tools = await mcp_client.get_tools()
    
    # 只保留需要用的部份
    gmail_tools = [t for t in all_mcp_tools if t.name in ALLOWED_GMAIL_TOOLS]
    print(f"[Agent] Loaded {len(gmail_tools)} Gmail tools: {[t.name for t in gmail_tools]}")
    
    
    # 建立GPT-4o然後將上面選出來的工具提供給模型
    llm = ChatOpenAI(model="gpt-4o", temperature=0).bind_tools(gmail_tools)
    
    # 定義我們這個功能裡面的子節點
    
    # 推理節點，將input抓下來後判斷是否要呼叫工具
    def agent_node(state: AgentState) -> AgentState:
        print(f"[Thinking] GPT-4o reasoning... (messages in state: {len(state['messages'])})")
        #將前面設定的prompt加在輸入前面送給llm
        response = llm.invoke([SYSTEM_PROMPT] + list(state["messages"]))
        if response.tool_calls:
            for tc in response.tool_calls:
                print(f"[Tool Call] → {tc['name']}({tc['args']})")
        else:
            print("[Final] GPT-4o returning final answer")
        
        return {"messages": [response]}
    
    # 條件節點，判斷是否要繼續執行或結束
    def should_continue(state: AgentState) -> str:
        
        # 判斷方式是透過最後一則訊息的文字有沒有帶有"tool_calls"來決定要不要繼續
        last_message = state["messages"][-1]
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "tools"  
        return END          
    
    # 建立好節點後以下就可以開始建立工作流（圖）
    
    # 建立 StateGraph，AgentState 結構
    graph = StateGraph(AgentState)
    
    #兩個節點，agent跟工具
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(gmail_tools))
    
    # start點
    graph.set_entry_point("agent")
    
    # 條件控制
    graph.add_conditional_edges("agent", should_continue)
    
    # tools好了之後要回agent
    graph.add_edge("tools", "agent")
    
    return graph.compile()


# 自動處理的模式（1），不需要互動，執行後會自動跑mail的流程
async def run_auto_refund_processing(agent) -> None:
    
    print("\n" + "="*60)
    print("Auto-processing refund & return emails...")
    print("="*60 + "\n")
    
    # 透過呼叫人類指令來觸發agent
    initial_message = HumanMessage(content=
        "Please execute the complete 6-step refund email processing workflow: "
        "search for all unread refund/return/complaint emails, read each one, "
        "classify them, compose appropriate replies using the templates, "
        "send threaded replies, and provide a final summary report."
    )
    
    # 執行agent並等待
    result = await agent.ainvoke({"messages": [initial_message]})
    
    # 取出最後一則AIMessage作為最終報告
    final_message = result["messages"][-1]
    print("\n" + "="*60)
    print("Agent Summary Report:")
    print("="*60)
    print(final_message.content)


# 互動對話模式，可以持續對話來輸入指令，並且會紀錄上一次對話來紀錄歷史資訊
async def run_interactive_chat(agent) -> None:
    
    print("\n" + "="*60)
    print("Refund Agent - Interactive Mode")
    print("Type 'exit' or 'quit' to stop")
    print("="*60 + "\n")
    
    # 完整對話歷史
    history = []
    
    while True:
        user_input = input("You: ").strip()
        
        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            print("Session ended.")
            break
        
        # 把使用者輸入加入歷史
        history.append(HumanMessage(content=user_input))
        
        # 把完整歷史傳給Agent
        result = await agent.ainvoke({"messages": history})
        
        # 取出最後一則回應
        ai_response = result["messages"][-1]
        print(f"\nAgent: {ai_response.content}\n")
        
        # 把回應也加入歷史，供下一輪參考
        history.append(ai_response)


# 確認環境變數用的，確保設定沒有問題
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
    # 先驗證環境變數
    if not validate_env():
        return
    
    print("Starting Refund Email Agent...")
    print("Connecting to Google Workspace MCP Server...")
    
    # call api
    mcp_client = MultiServerMCPClient(WORKSPACE_MCP_CONFIG)
    
    # 建立agent
    agent = await build_agent(mcp_client)
    print("Agent ready!\n")
    
    # 選擇執行模式
    print("Select mode:")
    print("  1. Auto mode - automatically process all refund emails")
    print("  2. Interactive mode - chat with the agent")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == "1":
        await run_auto_refund_processing(agent)
    elif choice == "2":
        await run_interactive_chat(agent)
    else:
        print("Invalid choice. Running auto mode by default.")
        await run_auto_refund_processing(agent)


if __name__ == "__main__":
    asyncio.run(main())

