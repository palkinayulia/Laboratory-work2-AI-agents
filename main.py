# -*- coding: utf-8 -*-
import asyncio
import json
import os
from pathlib import Path
from typing import TypedDict

from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver


# ============================================================
# 1. НАСТРОЙКИ
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "input_data.json"
METHODOLOGY_FILE = BASE_DIR / "methodology.txt"
MEMORY_FILE = BASE_DIR / "memory.sqlite"
OUTPUT_FILE = BASE_DIR / "final_report.txt"

MODEL_NAME = "qwen3:8b"
EMBEDDING_MODEL = "nomic-embed-text"
THREAD_ID = "management-report-demo"

llm = ChatOllama(model=MODEL_NAME, temperature=0)


# ============================================================
# 2. СОСТОЯНИЕ LANGGRAPH
# ============================================================

class ReportState(TypedDict, total=False):
    data: dict
    indicators: dict
    previous_data: dict
    previous_indicators: dict
    previous_report: str
    comparison: str
    analysis: str
    methodology_context: str
    verified_analysis: str
    final_report: str


# ============================================================
# 3. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================

def response_text(response) -> str:
    """Преобразует ответ языковой модели в текст."""
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)


def extract_mcp_text(result) -> str:
    """Извлекает текст из распространённых форматов ответа MCP."""
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        if "text" in result:
            return str(result["text"])
        if "content" in result:
            return extract_mcp_text(result["content"])
        return json.dumps(result, ensure_ascii=False)
    if isinstance(result, list):
        parts = []
        for item in result:
            if isinstance(item, dict):
                if "text" in item:
                    parts.append(str(item["text"]))
                elif "content" in item:
                    parts.append(extract_mcp_text(item["content"]))
                else:
                    parts.append(str(item))
            elif hasattr(item, "text"):
                parts.append(str(item.text))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    if hasattr(result, "content"):
        return extract_mcp_text(result.content)
    return str(result)


def format_number(value: float) -> str:
    """Выводит числа без лишних десятичных знаков."""
    return f"{value:,.2f}".replace(",", " ").rstrip("0").rstrip(".")


def calculate_indicators(data: dict) -> dict:
    """Рассчитывает отклонения программно, а не языковой моделью."""
    plan = float(data["plan"])
    fact = float(data["fact"])
    deviation = fact - plan
    percent = deviation / plan * 100 if plan != 0 else None
    return {
        "plan": plan,
        "fact": fact,
        "deviation": deviation,
        "deviation_percent": percent,
        "unit": str(data.get("unit", "единиц")),
    }


def make_comparison(previous_data, previous_indicators, data, indicators) -> str:
    """Готовит проверяемое сравнение с данными предыдущего запуска."""
    if not isinstance(previous_data, dict) or not isinstance(previous_indicators, dict):
        return "Предыдущие данные для сравнения отсутствуют."

    # Сравниваем только одинаковые подразделение и период.
    if (
        previous_data.get("department") != data.get("department")
        or previous_data.get("period") != data.get("period")
    ):
        return (
            "Предыдущий запуск относится к другому подразделению или периоду. "
            "Сравнение показателей не выполняется."
        )

    previous_fact = float(previous_indicators["fact"])
    current_fact = float(indicators["fact"])
    delta = current_fact - previous_fact
    sign = "+" if delta > 0 else ""

    previous_percent = previous_indicators.get("deviation_percent")
    current_percent = indicators.get("deviation_percent")
    if previous_percent is None or current_percent is None:
        percent_change = (
            "Процентные отклонения нельзя сравнить, поскольку план одного "
            "из запусков равен нулю."
        )
    else:
        percent_delta = float(current_percent) - float(previous_percent)
        percent_sign = "+" if percent_delta > 0 else ""
        percent_change = (
            f"Изменение процентного отклонения от плана: "
            f"{percent_sign}{format_number(percent_delta)} процентного пункта "
            f"(было {format_number(previous_percent)}%, стало "
            f"{format_number(current_percent)}%)."
        )

    return (
        f"Предыдущий запуск: факт {format_number(previous_fact)} "
        f"{indicators['unit']}. Текущий запуск: факт "
        f"{format_number(current_fact)} {indicators['unit']}. "
        f"Изменение факта: {sign}{format_number(delta)} "
        f"{indicators['unit']}. {percent_change}"
    )


# ============================================================
# 4. MCP: ПОЛУЧЕНИЕ ВХОДНЫХ ДАННЫХ
# ============================================================

async def read_input_via_mcp() -> dict:
    """Читает input_data.json через файловый MCP-сервер."""
    appdata = os.environ.get("APPDATA")
    if not appdata:
        raise RuntimeError("Не найдена переменная окружения APPDATA.")

    mcp_command = Path(appdata) / "npm" / "mcp-server-filesystem.cmd"
    if not mcp_command.exists():
        raise FileNotFoundError(f"Не найден MCP-сервер: {mcp_command}")

    client = MultiServerMCPClient({
        "filesystem": {
            "transport": "stdio",
            "command": str(mcp_command),
            "args": [str(BASE_DIR)],
        }
    })

    tools = await client.get_tools()
    read_tool = next((tool for tool in tools if tool.name == "read_text_file"), None)
    if read_tool is None:
        read_tool = next((tool for tool in tools if tool.name == "read_file"), None)
    if read_tool is None:
        raise RuntimeError("MCP-сервер не предоставил инструмент чтения файлов.")

    result = await read_tool.ainvoke({"path": str(DATA_FILE)})
    data_text = extract_mcp_text(result).strip()

    try:
        data = json.loads(data_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Ответ MCP не удалось разобрать как JSON:\n{data_text}") from exc

    if not isinstance(data, dict):
        raise ValueError("Входные данные должны быть JSON-объектом.")

    required = ["department", "period", "plan", "fact"]
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError("В input_data.json отсутствуют поля: " + ", ".join(missing))

    try:
        float(data["plan"])
        float(data["fact"])
    except (TypeError, ValueError) as exc:
        raise ValueError("План и факт должны быть числами.") from exc

    print(f"✓ MCP: данные получены ({data['department']}, {data['period']})")
    return data


# ============================================================
# 5. RAG: ПОИСК ПО МЕТОДИКЕ
# ============================================================

def build_vector_store():
    """Создаёт векторное хранилище по methodology.txt."""
    if not METHODOLOGY_FILE.exists():
        raise FileNotFoundError(f"Не найден файл методики: {METHODOLOGY_FILE}")

    text = METHODOLOGY_FILE.read_text(encoding="utf-8")
    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
    documents = splitter.create_documents(
        [text],
        metadatas=[{"source": METHODOLOGY_FILE.name}],
    )
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    return FAISS.from_documents(documents, embeddings)


def retrieve_methodology(vector_store, query: str) -> str:
    """Возвращает несколько релевантных фрагментов методики."""
    documents = vector_store.similarity_search(query, k=3)
    if not documents:
        return "Релевантные фрагменты методики не найдены."
    return "\n\n".join(
        f"Источник: {doc.metadata.get('source', 'methodology.txt')}\n{doc.page_content}"
        for doc in documents
    )


# ============================================================
# 6. АГЕНТ 1: АНАЛИЗ ПОКАЗАТЕЛЕЙ
# ============================================================

async def analyst_node(state: ReportState) -> dict:
    data = state["data"]
    indicators = state["indicators"]
    percent = indicators["deviation_percent"]
    percent_text = (
        f"{format_number(percent)}%"
        if percent is not None
        else "не рассчитывается, так как план равен нулю"
    )

    prompt = f"""
Ты первый агент системы управленческой отчётности.
Кратко опиши результаты, используя только известные данные.

Данные:
{json.dumps(data, ensure_ascii=False, indent=2)}

Результаты программного расчёта:
Абсолютное отклонение: {format_number(indicators['deviation'])}
Процентное отклонение: {percent_text}

Правила:
- Не пересчитывай показатели самостоятельно.
- Не выдумывай причины отклонения.
- Не утверждай, что положительное отклонение обязательно хорошо.
- Не добавляй сведения о бюджете, нагрузке, качестве или ошибках,
  если их нет в исходных данных.
- Укажи, что для оценки результата необходимо понимать смысл показателя.
- Ответ не более 100 слов.
"""
    analysis = response_text(await llm.ainvoke(prompt)).strip()
    print("✓ Агент 1: предварительный анализ выполнен")
    return {"analysis": analysis}


# ============================================================
# 7. АГЕНТ 2: ПРОВЕРКА ПО МЕТОДИКЕ С ИСПОЛЬЗОВАНИЕМ RAG
# ============================================================

async def methodology_node(state: ReportState, vector_store) -> dict:
    data = state["data"]
    indicators = state["indicators"]
    query = (
        "Формулы абсолютного и процентного отклонения, интерпретация "
        "отклонений, ограничения анализа и рекомендации для управленческой отчётности."
    )
    context = retrieve_methodology(vector_store, query)

    prompt = f"""
Ты второй агент. Проверь предварительный анализ первого агента
на соответствие методике. Не составляй полный итоговый отчёт.

Исходные данные:
{json.dumps(data, ensure_ascii=False, indent=2)}

Расчёты программы:
{json.dumps(indicators, ensure_ascii=False, indent=2)}

Анализ первого агента:
{state.get("analysis", "")}

Фрагменты методики, найденные через RAG:
{context}

Проверь:
1. Отклонение = факт - план.
2. Процент = (факт - план) / план * 100%, если план не равен нулю.
3. Положительное отклонение не всегда означает улучшение.
4. Причины отклонения нельзя утверждать без подтверждений.
5. Не добавляй факты, которых нет в исходных данных.

Верни краткий результат проверки: ошибки, если есть, и важные ограничения.
Не повторяй подробно все формулы. Максимум 100 слов.
"""
    verified = response_text(await llm.ainvoke(prompt)).strip()
    print("✓ Агент 2: анализ проверен по методике RAG")
    return {"methodology_context": context, "verified_analysis": verified}


# ============================================================
# 8. АГЕНТ 3: ИТОГОВЫЙ ОТЧЁТ С ИСПОЛЬЗОВАНИЕМ ПАМЯТИ
# ============================================================

async def report_writer_node(state: ReportState) -> dict:
    data = state["data"]
    indicators = state["indicators"]
    percent = indicators["deviation_percent"]
    unit = indicators["unit"]
    percent_text = (
        f"{format_number(percent)}%"
        if percent is not None
        else "не рассчитывается (план равен нулю)"
    )
    deviation = indicators["deviation"]
    deviation_sign = "+" if deviation > 0 else ""
    previous_report = state.get("previous_report", "")
    if not previous_report:
        previous_report = "Предыдущий отчёт отсутствует."
    comparison = state.get("comparison", "Сравнение с предыдущим запуском недоступно.")

    prompt = f"""
Ты третий агент системы. Подготовь краткий профессиональный
управленческий отчёт на русском языке.

Используй только факты из исходных данных и расчётов программы.
Не считай выводы предыдущих агентов дополнительными источниками фактов.

Исходные данные:
Подразделение: {data['department']}
Период: {data['period']}
План: {format_number(indicators['plan'])} {unit}
Факт: {format_number(indicators['fact'])} {unit}
Абсолютное отклонение: {deviation_sign}{format_number(deviation)} {unit}
Процентное отклонение: {percent_text}

Предварительный анализ:
{state.get("analysis", "")}

Проверка по методике:
{state.get("verified_analysis", "")}

Методика RAG:
{state.get("methodology_context", "")}

Предыдущий отчёт из памяти:
{previous_report}

Сравнение, рассчитанное программой:
{comparison}

Составь отчёт по структуре:

# Управленческий отчёт
**Подразделение:** ...
**Период:** ...

## Результаты
Кратко укажи план, факт, абсолютное и процентное отклонения.

## Интерпретация
В 1–2 предложениях объясни направление отклонения. Если смысл показателя
неизвестен, не называй результат успешным или неуспешным.

## Сравнение с предыдущим запуском
Если программа предоставила фактические данные для сравнения, кратко
укажи изменение факта и процентного отклонения. Это сравнение запусков,
а не обязательно разных периодов. Если сравнение недоступно, укажи это
одной короткой фразой.

## Ограничения
Укажи реальные ограничения исходных данных. Не выдумывай причины.

## Рекомендации
Дай 2–3 конкретных действия, связанных с ситуацией.

Ограничения:
- Не придумывай причины и факты.
- Не повторяй показатели и одну мысль в нескольких разделах.
- Не добавляй детали о бюджете, нагрузке, ошибках и качестве без данных.
- Максимум 250 слов.
- Не добавляй вступлений и комментариев от лица ИИ.
"""
    report = response_text(await llm.ainvoke(prompt)).strip()
    print("✓ Агент 3: итоговый отчёт подготовлен с учётом памяти")
    return {"final_report": report}


# ============================================================
# 9. ГРАФ LANGGRAPH
# ============================================================

def build_graph(vector_store):
    graph = StateGraph(ReportState)
    graph.add_node("analyst", analyst_node)

    async def methodology_agent(state: ReportState) -> dict:
        return await methodology_node(state, vector_store)

    graph.add_node("methodology_expert", methodology_agent)
    graph.add_node("report_writer", report_writer_node)
    graph.add_edge(START, "analyst")
    graph.add_edge("analyst", "methodology_expert")
    graph.add_edge("methodology_expert", "report_writer")
    graph.add_edge("report_writer", END)
    return graph


# ============================================================
# 10. ЗАПУСК, ПАМЯТЬ И СОХРАНЕНИЕ ОТЧЁТА
# ============================================================

async def main():
    print("\nСИСТЕМА УПРАВЛЕНЧЕСКОЙ ОТЧЁТНОСТИ")
    print("-" * 42)

    data = await read_input_via_mcp()
    indicators = calculate_indicators(data)

    print("Подготовка RAG...")
    vector_store = build_vector_store()
    print("✓ RAG: методика загружена")

    graph = build_graph(vector_store)
    config = {"configurable": {"thread_id": THREAD_ID}}

    print("Подключение памяти SQLite...")
    async with AsyncSqliteSaver.from_conn_string(str(MEMORY_FILE)) as checkpointer:
        app = graph.compile(checkpointer=checkpointer)

        previous_state = await app.aget_state(config)
        previous_values = previous_state.values if previous_state.values else {}

        previous_report = previous_values.get("final_report", "")
        previous_data = previous_values.get("data")
        previous_indicators = previous_values.get("indicators")

        comparison = make_comparison(
            previous_data,
            previous_indicators,
            data,
            indicators,
        )

        if previous_report:
            print("✓ Память: предыдущий отчёт восстановлен")
            print("✓ Память: данные предыдущего запуска подготовлены к сравнению")
        else:
            print("✓ Память: первый запуск, предыдущего отчёта нет")

        print("\nПоследовательная обработка:")
        result = await app.ainvoke(
            {
                "data": data,
                "indicators": indicators,
                "previous_data": previous_data or {},
                "previous_indicators": previous_indicators or {},
                "previous_report": previous_report,
                "comparison": comparison,
            },
            config=config,
        )

    report = result.get("final_report", "").strip()
    if not report:
        raise RuntimeError("Агент 3 не сформировал итоговый отчёт.")

    OUTPUT_FILE.write_text(report + "\n", encoding="utf-8")

    print("\n" + "=" * 42)
    print("ИТОГОВЫЙ ОТЧЁТ")
    print("=" * 42)
    print(report)
    print("\n" + "-" * 42)
    print(f"Файл: {OUTPUT_FILE}")
    print("✓ Работа системы завершена")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as exc:
        print(f"\nОшибка: {type(exc).__name__}: {exc}")
        raise
