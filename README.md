# AI Workspace Agent Suite

## 環境
- Python 3.11
- OpenAI GPT-4o
- Google Workspace MCP v1.21.0

## 步驟

### 安裝套件
```bash
git clone https://github.com/taylorwilsdon/google_workspace_mcp
cd google_workspace_mcp
pip install uv
uv tool install .
pip install workspace-mcp
pip install langgraph langchain-openai langchain-mcp-adapters langchain-core google-api-python-client
```

### 設定 .env
GOOGLE_OAUTH_CLIENT_ID=你的_client_id
GOOGLE_OAUTH_CLIENT_SECRET=你的_client_secret
OAUTHLIB_INSECURE_TRANSPORT=1
OPENAI_API_KEY=你的_openai_api_key
USER_GOOGLE_EMAIL=你的gmail@gmail.com
### 執行
```bash
python main.py
```

## 系統架構
- agent_node：GPT-4o 推理，決定呼叫哪個工具
- tool_node：執行 Gmail 或 Calendar 工具
- should_continue：判斷是否繼續 ReAct loop

## Agent 說明
- Refund Email Agent：自動搜尋、分類、回覆 Gmail 退款信件
- Calendar Agent：透過自然語言查詢與管理 Google Calendar
