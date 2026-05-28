# main.py
# AI Workspace Agent Suite - 統一入口
# 執行完任一 Agent 後會返回主選單重新選擇

import subprocess
import sys
import os


def print_banner():
    """印出專案標題"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║              AI Workspace Agent Suite                        ║
║      Google Workspace MCP + LangGraph + GPT-4o              ║
╚══════════════════════════════════════════════════════════════╝
    """)


def clear_ports():
    """清除可能被佔用的 MCP server ports，避免每次都跑到 8001"""
    for port in [8000, 8001, 8002, 8003]:
        try:
            subprocess.run(
                f"kill $(lsof -t -i:{port}) 2>/dev/null",
                shell=True,
                capture_output=True
            )
        except:
            pass


def main():
    print_banner()

    # 取得目前資料夾路徑
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 持續顯示選單直到使用者選擇退出
    while True:
        print("Available Agents:")
        print("─" * 40)
        print("  1. Refund Email Agent")
        print("     → Auto-process Gmail refund/return emails")
        print()
        print("  2. Calendar Agent")
        print("     → Manage Google Calendar with natural language")
        print()
        print("  0. Exit")
        print("─" * 40)

        choice = input("\nSelect agent (0, 1 or 2): ").strip()

        if choice == "1":
            print("\nCleaning up ports...")
            clear_ports()  # 執行前先清除佔用的 port
            print("Starting Refund Email Agent...\n")
            subprocess.run(
                [sys.executable, os.path.join(current_dir, "refund_agent.py")]
            )
            print("\n" + "="*60)
            print("Refund Email Agent finished. Returning to main menu...")
            print("="*60 + "\n")

        elif choice == "2":
            print("\nCleaning up ports...")
            clear_ports()  # 執行前先清除佔用的 port
            print("Starting Calendar Agent...\n")
            subprocess.run(
                [sys.executable, os.path.join(current_dir, "calendar_agent.py")]
            )
            print("\n" + "="*60)
            print("Calendar Agent finished. Returning to main menu...")
            print("="*60 + "\n")

        elif choice == "0":
            clear_ports()  # 退出前也清除
            print("\nGoodbye!")
            break

        else:
            print("\nInvalid choice. Please enter 0, 1 or 2.\n")


if __name__ == "__main__":
    main()
