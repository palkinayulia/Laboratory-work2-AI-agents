
import asyncio
import os

from langchain_mcp_adapters.client import MultiServerMCPClient


async def main():
    client = MultiServerMCPClient(
        {
            "filesystem": {
                "transport": "stdio",
                "command": os.path.join(
                    os.environ["APPDATA"],
                    "npm",
                    "mcp-server-filesystem.cmd",
                ),
                "args": [r"C:\lab2_ai_agents"],
            }
        }
    )

    tools = await client.get_tools()

    print("MCP подключен успешно!")
    print("\nИнструменты, полученные от сервера:")

    for tool in tools:
        print(f"- {tool.name}: {tool.description[:100]}")


if __name__ == "__main__":
    asyncio.run(main())
