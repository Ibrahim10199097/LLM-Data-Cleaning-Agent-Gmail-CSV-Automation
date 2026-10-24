"""
LLM-driven CSV cleaning agent (LangGraph)
=========================================

Replaces the fixed `clean_csv()` steps with an agent that:

  1. receives a PROFILE of the file (schema, null counts, samples),
  2. PLANS which cleaning tools to call, on which columns, and in what order,
  3. OBSERVES each tool result (tools can refuse unsafe actions),
  4. repeats until it calls `finish_cleaning(summary)`.

Safety / guardrails
-------------------
* Only whitelisted tools can run - the LLM never executes free-form code.
* The original bytes are never modified; the agent works on a copy.
* Tools validate their own inputs (unknown column, <90% numeric, identifiers
  with leading zeros, dropping a column that is mostly filled, ...).
* Hard step limit (MAX_AGENT_STEPS).
* If the agent fails, makes no changes, or removes more than half of the rows,
  the fixed pipeline is used instead (`fallback`), so an email is never lost.

Requirements:
    pip install pandas langgraph langchain-core
    # plus ONE provider package:
    pip install langchain-google-genai      # Gemini (has a free tier)
    pip install langchain-anthropic         # Claude
    pip install langchain-openai            # OpenAI

Environment:
    LLM_PROVIDER = gemini | anthropic | openai     (default: gemini)
    LLM_MODEL    = optional model name override
    GOOGLE_API_KEY / ANTHROPIC_API_KEY / OPENAI_API_KEY
"""

import io
import logging
import os
import re
from typing import Annotated, Literal, TypedDict

import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

log = logging.getLogger(__name__)

MAX_AGENT_STEPS = 30          # max LLM <-> tool round trips
MAX_PROFILE_COLUMNS = 60      # keeps the prompt small for very wide files
CONVERT_THRESHOLD = 0.90      # share of values that must parse to convert a column
DROP_COLUMN_MIN_MISSING = 0.50  # tool refuses to drop a column with less missing

MISSING_TOKENS = {"", "na", "n/a", "null", "none", "-", "?", "blank"}
IDENTIFIER_WORDS = ["id", "phone", "mobile", "zip", "postal", "postcode", "sku", "code"]


# ============================================================
# Small helpers (same behaviour as the fixed pipeline)
# ============================================================

def normalize_column_name(column) -> str:
    column = str(column).strip().lower()
    column = re.sub(r"[^a-zA-Z0-9]+", "_", column)
    column = re.sub(r"_+", "_", column)
    return column.strip("_")


def clean_text(value):
    if pd.isna(value):
        return pd.NA
    value = re.sub(r"\s+", " ", str(value)).strip()
    return pd.NA if value.lower() in MISSING_TOKENS else value


def looks_like_identifier(column_name) -> bool:
    name = str(column_name).lower()
    return any(word in name for word in IDENTIFIER_WORDS)


def has_leading_zeros(series: pd.Series) -> bool:
    values = series.dropna().astype("string")
    return bool(values.str.match(r"^0\d+$").any())


# ============================================================
# Profile given to the LLM
# ============================================================

def profile_dataframe(df: pd.DataFrame, max_columns: int = MAX_PROFILE_COLUMNS) -> str:
    lines = [f"rows={len(df)}, columns={len(df.columns)}, "
             f"exact_duplicate_rows={int(df.duplicated().sum())}"]

    for column in list(df.columns)[:max_columns]:
        s = df[column]
        raw_missing = int(
            s.isna().sum()
            + s.astype("string").str.strip().str.lower().isin(MISSING_TOKENS).sum()
        )
        samples = s.dropna().astype("string").str.strip()
        samples = samples[~samples.str.lower().isin(MISSING_TOKENS)]
        samples = samples.drop_duplicates().head(6).tolist()
        lines.append(
            f"- {column!r}: dtype={s.dtype}, missing_or_placeholder={raw_missing}, "
            f"unique={int(s.nunique(dropna=True))}, samples={samples}"
        )

    if len(df.columns) > max_columns:
        lines.append(f"... {len(df.columns) - max_columns} more columns not shown")
    return "\n".join(lines)


# ============================================================
# Cleaning session + tools
# ============================================================

class CleaningSession:
    """Holds the DataFrame being cleaned and an audit log of every action."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.rows_before = len(df)
        self.columns_before = len(df.columns)
        self.actions: list[dict] = []
        self.numeric_columns: list[str] = []
        self.date_columns: list[str] = []
        self.date_sources: dict[str, pd.Series] = {}  # text before the first date conversion
        self.dropped_columns: list[str] = []
        self.duplicates_removed = 0
        self.empty_rows_removed = 0
        self.empty_columns_removed = 0
        self.missing_before = int(
            df.apply(lambda col: col.map(clean_text)).isna().sum().sum()
        )
        self.finished = False
        self.summary = ""

    def log_action(self, tool_name, args, result, reason):
        self.actions.append(
            {"tool": tool_name, "args": args, "result": result, "reason": reason}
        )

    def resolve(self, name: str) -> str:
        columns = list(self.df.columns)
        if name in columns:
            return name
        normalized = normalize_column_name(name)
        if normalized in columns:
            return normalized
        for column in columns:
            if str(column).strip().lower() == str(name).strip().lower():
                return column
        raise ValueError(f"Unknown column {name!r}. Available columns: {columns}")


def make_tools(session: CleaningSession):
    """Build the whitelisted tool set bound to one session."""

    def guarded(tool_name, args, reason, fn):
        try:
            result = fn()
        except Exception as exc:  # tool errors go back to the LLM, not up the stack
            result = f"ERROR: {exc}"
        session.log_action(tool_name, args, result, reason)
        return result

    @tool
    def profile_data() -> str:
        """Show the CURRENT state of the data (columns, dtypes, missing counts, samples).
        Call this to check your work after several changes."""
        return profile_dataframe(session.df)

    @tool
    def normalize_column_names(reason: str) -> str:
        """Rename all columns to lower_snake_case (e.g. ' Order Amount ($) ' -> 'order_amount').
        Afterwards, use the NEW names in other tools."""
        def run():
            old = list(session.df.columns)
            session.df.columns = [normalize_column_name(c) for c in old]
            if session.df.columns.duplicated().any():
                session.df.columns = old
                raise ValueError("Normalizing would create duplicate column names; reverted.")
            return f"Renamed columns: {dict(zip(old, session.df.columns))}"
        return guarded("normalize_column_names", {}, reason, run)

    @tool
    def clean_text_values(reason: str) -> str:
        """Trim and collapse whitespace in every cell and convert placeholders
        ('', NA, N/A, null, none, -, ?, blank) to real missing values."""
        def run():
            for column in session.df.columns:
                session.df[column] = session.df[column].map(clean_text)
            return "Whitespace trimmed; placeholders converted to missing."
        return guarded("clean_text_values", {}, reason, run)

    @tool
    def drop_empty_rows_and_columns(reason: str) -> str:
        """Remove rows and columns that are completely empty (run clean_text_values first)."""
        def run():
            rows = int(session.df.isna().all(axis=1).sum())
            session.df = session.df.dropna(how="all")
            cols = int(session.df.isna().all(axis=0).sum())
            session.df = session.df.dropna(axis=1, how="all")
            session.empty_rows_removed += rows
            session.empty_columns_removed += cols
            return f"Removed {rows} empty rows and {cols} empty columns."
        return guarded("drop_empty_rows_and_columns", {}, reason, run)

    @tool
    def drop_duplicate_rows(reason: str) -> str:
        """Remove rows that are exact duplicates of an earlier row."""
        def run():
            n = int(session.df.duplicated().sum())
            session.df = session.df.drop_duplicates()
            session.duplicates_removed += n
            return f"Removed {n} duplicate rows."
        return guarded("drop_duplicate_rows", {}, reason, run)

    @tool
    def convert_to_numeric(column: str, reason: str) -> str:
        """Convert a column to numbers. Strips currency symbols/codes, commas and spaces
        (e.g. '$1,200.50', 'EGP 950'). Refuses if fewer than 90% of values parse, or if the
        column looks like an identifier (IDs, phone, zip, account numbers with leading zeros)."""
        def run():
            col = session.resolve(column)
            s = session.df[col]
            if looks_like_identifier(col) or has_leading_zeros(s):
                raise ValueError(
                    f"{col!r} looks like an identifier (name or leading zeros). "
                    "Keep it as text."
                )
            text = s.astype("string")
            stripped = text.str.replace(r"[^0-9.\-]", "", regex=True).replace("", pd.NA)
            converted = pd.to_numeric(stripped, errors="coerce")
            non_empty = int(s.notna().sum())
            if non_empty == 0:
                raise ValueError(f"{col!r} has no values.")
            rate = converted.notna().sum() / non_empty
            if rate < CONVERT_THRESHOLD:
                bad = text[converted.isna() & s.notna()].head(3).tolist()
                raise ValueError(
                    f"Only {rate:.0%} of {col!r} parses as numeric (need 90%). "
                    f"Examples that failed: {bad}. Not converted."
                )
            non_null = converted.dropna()
            if (non_null == non_null.round()).all():
                converted = converted.astype("Int64")
            session.df[col] = converted
            if col not in session.numeric_columns:
                session.numeric_columns.append(col)
            return f"Converted {col!r} to numeric ({rate:.0%} parsed)."
        return guarded("convert_to_numeric", {"column": column}, reason, run)

    @tool
    def convert_to_date(column: str, day_first: bool, reason: str) -> str:
        """Convert a column to ISO dates (YYYY-MM-DD). Set day_first=True for dd/mm/yyyy,
        False for mm/dd/yyyy. Decide from the samples: if any value like '25/12/2024' or
        '13/02/2024' appears, the first number is the DAY. If every value is ambiguous,
        prefer day_first=True and say so in your reason. Refuses if <90% parse."""
        def run():
            col = session.resolve(column)
            # Always parse from the text as it was before the first conversion, so
            # retrying with a different day_first value starts from the original dates.
            session.date_sources.setdefault(col, session.df[col].copy())
            s = session.date_sources[col].reindex(session.df.index)
            parsed = pd.to_datetime(
                s.astype("string"), errors="coerce", dayfirst=day_first, format="mixed"
            )
            non_empty = int(s.notna().sum())
            if non_empty == 0:
                raise ValueError(f"{col!r} has no values.")
            rate = parsed.notna().sum() / non_empty
            if rate < CONVERT_THRESHOLD:
                bad = s[parsed.isna() & s.notna()].head(3).tolist()
                raise ValueError(
                    f"Only {rate:.0%} of {col!r} parses as a date with day_first={day_first}. "
                    f"Examples that failed: {bad}. Try the other day_first value or leave as text."
                )
            session.df[col] = parsed.dt.strftime("%Y-%m-%d")
            if col not in session.date_columns:
                session.date_columns.append(col)
            return f"Converted {col!r} to ISO dates (day_first={day_first}, {rate:.0%} parsed)."
        return guarded("convert_to_date", {"column": column, "day_first": day_first}, reason, run)

    @tool
    def standardize_case(column: str, case: Literal["lower", "upper", "title"], reason: str) -> str:
        """Make text categories consistent (e.g. 'cairo', 'CAIRO ', 'Cairo' -> 'Cairo' with
        case='title'). Only for text columns of categories - not names of people or free text."""
        def run():
            col = session.resolve(column)
            if pd.api.types.is_numeric_dtype(session.df[col]) or col in session.date_columns:
                raise ValueError(f"{col!r} is not a text column.")
            s = session.df[col].astype("string").str.strip()
            session.df[col] = getattr(s.str, case)()
            return f"Standardized {col!r} to {case} case ({session.df[col].nunique()} unique values)."
        return guarded("standardize_case", {"column": column, "case": case}, reason, run)

    @tool
    def fill_missing_values(
        column: str,
        strategy: Literal["median", "mean", "mode", "unknown", "leave"],
        reason: str,
    ) -> str:
        """Handle missing values in ONE column. Numeric: median | mean | mode | leave.
        Text: mode | unknown | leave. Date columns: leave. Use 'leave' when filling would
        invent data (IDs, emails, free text, dates)."""
        def run():
            col = session.resolve(column)
            s = session.df[col]
            n_missing = int(s.isna().sum())
            if strategy == "leave" or n_missing == 0:
                return f"{col!r}: {n_missing} missing left as is."
            if col in session.date_columns:
                raise ValueError("Date columns can only be left as is.")
            if pd.api.types.is_numeric_dtype(s):
                if strategy == "unknown":
                    raise ValueError("'unknown' is for text columns; use median, mean or mode.")
                if s.notna().sum() == 0:
                    raise ValueError(f"{col!r} has no values to compute a {strategy} from.")
                value = {"median": s.median, "mean": s.mean}.get(strategy, lambda: s.mode().iloc[0])()
                if str(s.dtype) == "Int64" and float(value) != int(value):
                    s = s.astype("float64")
                session.df[col] = s.fillna(value)
            else:
                if strategy in ("median", "mean"):
                    raise ValueError(f"{strategy} needs a numeric column; use mode or unknown.")
                value = "Unknown" if strategy == "unknown" else s.mode().iloc[0]
                session.df[col] = s.fillna(value)
            return f"Filled {n_missing} missing in {col!r} with {strategy} ({value})."
        return guarded("fill_missing_values", {"column": column, "strategy": strategy}, reason, run)

    @tool
    def drop_column(column: str, reason: str) -> str:
        """Drop a column that is mostly empty and not useful. Refuses unless at least 50%
        of its values are missing."""
        def run():
            col = session.resolve(column)
            share = session.df[col].isna().mean()
            if share < DROP_COLUMN_MIN_MISSING:
                raise ValueError(
                    f"{col!r} is only {share:.0%} missing; refusing to drop it."
                )
            session.df = session.df.drop(columns=[col])
            session.dropped_columns.append(col)
            return f"Dropped {col!r} ({share:.0%} missing)."
        return guarded("drop_column", {"column": column}, reason, run)

    @tool
    def finish_cleaning(summary: str) -> str:
        """Call ONCE when the file is fully cleaned. `summary` is a short plain-English
        explanation for the file's owner of what you changed and why, and anything you
        deliberately left alone."""
        session.finished = True
        session.summary = summary
        session.log_action("finish_cleaning", {}, "done", "")
        return "Cleaning finished."

    return [
        profile_data,
        normalize_column_names,
        clean_text_values,
        drop_empty_rows_and_columns,
        drop_duplicate_rows,
        convert_to_numeric,
        convert_to_date,
        standardize_case,
        fill_missing_values,
        drop_column,
        finish_cleaning,
    ]


# ============================================================
# LLM + graph
# ============================================================

SYSTEM_PROMPT = """You are a careful data-cleaning agent. A user emailed you a CSV file.
Clean it by calling the provided tools - you cannot run code. You will receive a profile
of the raw data (every value is initially text).

Workflow: normalize_column_names -> clean_text_values -> drop_empty_rows_and_columns ->
drop_duplicate_rows -> then decide PER COLUMN: type conversion, category standardization,
missing-value handling. Finally call finish_cleaning with a short summary.

Policy:
- Never invent information. Do not fill IDs, emails, names, free text or dates.
- Keep identifiers (IDs, phone numbers, ZIP/postal codes, account numbers, SKUs) as TEXT so
  leading zeros survive. Never convert them to numbers.
- Numeric columns with missing values: fill with the median unless the samples suggest otherwise.
- Low-cardinality text categories: make the case consistent. Leave free text alone.
- Columns that are more than 70% missing: drop them instead of filling them.
- Dates: infer day-first vs month-first from the samples (a value like 25/12/2024 or 13/02/2024
  means day-first). If everything is ambiguous, prefer day-first and say so.
- If a tool answers ERROR, read the message and adapt; do not repeat the same call.
- If a file is already clean, change as little as possible.
- Every tool takes a `reason`: one short sentence explaining your decision.
- Do not make more than one tool call per column per step unless needed. Finish within 30 steps."""


def get_llm():
    """Create the chat model from environment variables (temperature 0)."""
    provider = os.getenv("LLM_PROVIDER", "gemini").lower()
    model = os.getenv("LLM_MODEL")

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model or "gemini-2.5-flash", temperature=0)
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model or "claude-haiku-5-5", temperature=0)
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model or "gpt-4o-mini", temperature=0)
    raise ValueError(f"Unknown LLM_PROVIDER {provider!r}")


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


def build_graph(llm, session: CleaningSession):
    tools = make_tools(session)
    llm_with_tools = llm.bind_tools(tools)

    def agent_node(state: AgentState):
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    def after_agent(state: AgentState):
        last = state["messages"][-1]
        return "tools" if getattr(last, "tool_calls", None) else END

    def after_tools(state: AgentState):
        return END if session.finished else "agent"

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools))
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", after_agent, {"tools": "tools", END: END})
    graph.add_conditional_edges("tools", after_tools, {"agent": "agent", END: END})
    return graph.compile()


# ============================================================
# Public entry point
# ============================================================

def _build_report(session: CleaningSession, mode: str) -> dict:
    return {
        "mode": mode,
        "rows_before": session.rows_before,
        "rows_after": len(session.df),
        "columns_before": session.columns_before,
        "columns_after": len(session.df.columns),
        "empty_rows_removed": session.empty_rows_removed,
        "empty_columns_removed": session.empty_columns_removed,
        "duplicates_removed": session.duplicates_removed,
        "missing_cells_before": session.missing_before,
        "missing_cells_after": int(session.df.isna().sum().sum()),
        "numeric_columns": session.numeric_columns,
        "date_columns": session.date_columns,
        "dropped_columns": session.dropped_columns,
        "agent_summary": session.summary,
        "actions": [a for a in session.actions if a["tool"] != "profile_data"],
    }


def clean_csv_with_agent(csv_bytes: bytes, fallback=None, llm=None):
    """
    Same contract as the fixed `clean_csv`:  returns (cleaned_csv_bytes, report).

    fallback: the fixed `clean_csv(csv_bytes)` function, used if the agent fails.
    llm:      optional chat model (defaults to get_llm()); handy for tests.
    """
    raw = pd.read_csv(io.BytesIO(csv_bytes), dtype=str, keep_default_na=False)
    session = CleaningSession(raw)

    try:
        llm = llm or get_llm()
        app = build_graph(llm, session)
        app.invoke(
            {
                "messages": [
                    SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(content="Raw data profile:\n" + profile_dataframe(raw)),
                ]
            },
            config={"recursion_limit": MAX_AGENT_STEPS * 2 + 2},
        )

        changed = [a for a in session.actions if a["tool"] != "profile_data"]
        if not changed:
            raise RuntimeError("agent made no changes")
        if len(session.df) < 0.5 * max(session.rows_before, 1):
            raise RuntimeError("agent removed more than half of the rows")
        if not session.finished:
            session.summary = "The agent stopped before writing a summary; see the action log."

        output = io.StringIO()
        session.df.to_csv(output, index=False)
        return output.getvalue().encode("utf-8"), _build_report(session, "agent")

    except Exception as exc:
        log.warning("Agent failed (%s).", exc)
        if fallback is None:
            raise
        log.warning("Using the fixed cleaning pipeline instead.")
        cleaned_bytes, report = fallback(csv_bytes)
        report.update(
            mode="fallback",
            agent_summary=f"The AI agent was unavailable ({exc}); standard cleaning rules were applied.",
            actions=[],
            dropped_columns=[],
        )
        return cleaned_bytes, report
