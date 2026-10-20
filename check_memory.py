# -*- coding: utf-8 -*-

import asyncio
from pathlib import Path

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver


async def main():
    db_path = Path(__file__).parent / "memory.sqlite"
    thread_id = "management-report-demo"

    async with AsyncSqliteSaver.from_conn_string(
        str(db_path)
    ) as checkpointer:
        config = {
            "configurable": {
                "thread_id": thread_id
            }
        }

        checkpoint = await checkpointer.aget_tuple(config)

        if checkpoint is None:
            print("Checkpoint не найден.")
            print("Thread ID:", thread_id)
            return

        print("=" * 55)
        print("ПРОВЕРКА ПАМЯТИ SQLITE")
        print("=" * 55)
        print("Checkpoint найден.")
        print("Thread ID:", thread_id)

        # Данные состояния графа
        channel_values = checkpoint.checkpoint.get(
            "channel_values", {}
        )

        print("\nСохранённые каналы:")
        if channel_values:
            for key in channel_values:
                print("-", key)
        else:
            print("Каналы состояния не найдены.")

        # Выводим содержимое важных полей состояния
        important_fields = [
            "data",
            "indicators",
            "previous_report",
            "analysis",
            "methodology_context",
            "verified_analysis",
            "final_report",
        ]

        print("\n" + "=" * 55)
        print("СОДЕРЖИМОЕ ПАМЯТИ")
        print("=" * 55)

        for field in important_fields:
            if field in channel_values:
                value = channel_values[field]
                print(f"\n--- {field} ---")

                text = str(value)
                if len(text) > 2500:
                    text = text[:2500] + "\n... [вывод сокращён]"

                print(text)

        print("\n" + "=" * 55)
        print("Проверка завершена.")


if __name__ == "__main__":
    asyncio.run(main())
