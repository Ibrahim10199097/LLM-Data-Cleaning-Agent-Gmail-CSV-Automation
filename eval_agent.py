"""
Evaluation: fixed pipeline vs LLM agent on messy CSV files with known answers.

Each test case is a small messy CSV plus CHECKS that encode the correct outcome
(e.g. "leading zeros survive", "25/12/2024 becomes 2024-12-25"). A case PASSES only
if every check passes.

Run:
    python eval_agent.py              # baseline (fixed pipeline) only - no API key needed
    python eval_agent.py --agent      # baseline + LLM agent (needs an API key)
    python eval_agent.py --agent --runs 3   # repeat the agent to measure consistency

Report numbers like "agent 8/8 vs fixed 4/8 test files" on your CV only after you have
actually run this.
"""

import argparse
import io
import sys
from unittest.mock import MagicMock

import pandas as pd


def _import_project():
    """Import the project module; stub Google libs if they are not installed
    (the cleaning functions do not use them)."""
    try:
        import email_csv_cleaning_automation as project
    except ImportError:
        for name in [
            "google", "google.auth", "google.auth.transport",
            "google.auth.transport.requests", "google.oauth2",
            "google.oauth2.credentials", "google_auth_oauthlib",
            "google_auth_oauthlib.flow", "googleapiclient", "googleapiclient.discovery",
        ]:
            sys.modules.setdefault(name, MagicMock())
        import email_csv_cleaning_automation as project
    return project


def to_csv_bytes(rows, columns):
    return pd.DataFrame(rows, columns=columns).to_csv(index=False).encode("utf-8")


def read(csv_bytes):
    return pd.read_csv(io.BytesIO(csv_bytes), dtype=str, keep_default_na=False)


# ------------------------------------------------------------
# Test cases: (name, csv_bytes, [(check_name, fn(df) -> bool)])
# ------------------------------------------------------------

def case_messy_headers_and_duplicates():
    rows = [
        ["A1", " Ali ", "100"], ["A2", "Mona", "200"], ["A2", "Mona", "200"],
        ["", "", ""], ["A3", "Omar", "300"],
    ]
    csv = to_csv_bytes(rows, [" Order ID ", "Customer  Name", "Amount ($)"])
    checks = [
        ("columns normalized", lambda d: list(d.columns) == ["order_id", "customer_name", "amount"]),
        ("duplicates + empty row removed", lambda d: len(d) == 3),
        ("whitespace trimmed", lambda d: "Ali" in d["customer_name"].tolist()),
    ]
    return "messy headers, duplicates, empty row", csv, checks


def case_leading_zero_identifiers():
    rows = [["00123", "01012345678", "50"], ["00456", "01198765432", "75"], ["00789", "01234567890", "60"]]
    csv = to_csv_bytes(rows, ["account_no", "contact", "balance"])
    checks = [
        ("account_no keeps leading zeros", lambda d: d["account_no"].tolist() == ["00123", "00456", "00789"]),
        ("contact keeps leading zeros", lambda d: d["contact"].iloc[0] == "01012345678"),
    ]
    return "identifiers with leading zeros", csv, checks


def case_day_first_dates():
    # The first value is ambiguous (05/03), so a parser that guesses month-first
    # from it breaks on the later 13/02 and 25/12 values.
    rows = [["1", "05/03/2024"], ["2", "13/02/2024"], ["3", "25/12/2023"], ["4", "01/11/2024"]]
    csv = to_csv_bytes(rows, ["n", "order_date"])
    expected = ["2024-03-05", "2024-02-13", "2023-12-25", "2024-11-01"]
    checks = [("dates read day-first as ISO", lambda d: d["order_date"].tolist() == expected)]
    return "day-first dates (dd/mm/yyyy)", csv, checks


def case_currency_text_numbers():
    rows = [["1", "$1,200.50"], ["2", "EGP 950"], ["3", "EGP 300"], ["4", "$75"], ["5", "EGP 1,000"]]
    csv = to_csv_bytes(rows, ["n", "price"])

    def numeric_ok(d):
        try:
            return abs(pd.to_numeric(d["price"]).sum() - 3525.5) < 0.01
        except Exception:
            return False

    return "currency symbols/codes in numbers", csv, [("price parsed to numbers", numeric_ok)]


def case_inconsistent_categories():
    rows = [["1", "Cairo"], ["2", "cairo "], ["3", "CAIRO"], ["4", "Giza"], ["5", "giza"], ["6", "GIZA"]]
    csv = to_csv_bytes(rows, ["n", "city"])
    checks = [("categories standardized to 2 values", lambda d: d["city"].nunique() == 2)]
    return "inconsistent category case", csv, checks


def case_mostly_empty_column():
    rows = [[str(i), f"c{i}", "note" if i == 0 else ("N/A" if i % 2 else "")] for i in range(10)]
    csv = to_csv_bytes(rows, ["n", "name", "notes"])
    checks = [("mostly-empty column not filled with 'Unknown'",
               lambda d: "notes" not in d.columns or (d["notes"] == "Unknown").mean() < 0.5)]
    return "column that is 90% empty", csv, checks


def case_numeric_missing_median():
    rows = [["1", "10"], ["2", "N/A"], ["3", "30"], ["4", ""], ["5", "50"]]
    csv = to_csv_bytes(rows, ["n", "score"])

    def ok(d):
        try:
            return pd.to_numeric(d["score"]).tolist() == [10, 30, 30, 30, 50]
        except Exception:
            return False

    return "numeric column with missing values", csv, [("missing filled with median (30)", ok)]


def case_already_clean():
    rows = [["1", "Ali", "10", "2024-01-05"], ["2", "Mona", "20", "2024-02-10"], ["3", "Omar", "30", "2024-03-15"]]
    csv = to_csv_bytes(rows, ["n", "name", "score", "joined"])
    checks = [
        ("rows unchanged", lambda d: len(d) == 3),
        ("names unchanged", lambda d: d["name"].tolist() == ["Ali", "Mona", "Omar"]),
        ("dates unchanged", lambda d: d["joined"].tolist() == ["2024-01-05", "2024-02-10", "2024-03-15"]),
        ("no 'Unknown' invented", lambda d: not (d == "Unknown").any().any()),
    ]
    return "already clean file (should not be damaged)", csv, checks


CASES = [
    case_messy_headers_and_duplicates, case_leading_zero_identifiers, case_day_first_dates,
    case_currency_text_numbers, case_inconsistent_categories, case_mostly_empty_column,
    case_numeric_missing_median, case_already_clean,
]


def run_case(csv_bytes, checks, cleaner):
    try:
        cleaned = cleaner(csv_bytes)
        df = read(cleaned)
    except Exception as exc:
        return False, [(name, False) for name, _ in checks], f"crashed: {exc}"
    results = []
    for name, fn in checks:
        try:
            results.append((name, bool(fn(df))))
        except Exception:
            results.append((name, False))
    return all(ok for _, ok in results), results, ""


def evaluate(label, cleaner, runs=1):
    print(f"\n=== {label} ===")
    total_pass = total = 0
    for _ in range(runs):
        for build in CASES:
            name, csv_bytes, checks = build()
            passed, results, error = run_case(csv_bytes, checks, cleaner)
            total += 1
            total_pass += passed
            failed = [n for n, ok in results if not ok]
            line = f"{'PASS' if passed else 'FAIL'}  {name}"
            if failed:
                line += f"   <- failed: {', '.join(failed)}"
            if error:
                line += f"   ({error})"
            print(line)
    print(f"{label}: {total_pass}/{total} test files fully correct")
    return total_pass, total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", action="store_true", help="also evaluate the LLM agent")
    parser.add_argument("--runs", type=int, default=1, help="repeat the agent N times")
    args = parser.parse_args()

    project = _import_project()
    evaluate("Fixed pipeline", lambda b: project.clean_csv(b)[0])

    if args.agent:
        from cleaning_agent import clean_csv_with_agent
        # fallback=None: a crash counts as a failure, so the score is the agent's own.
        evaluate("LLM agent", lambda b: clean_csv_with_agent(b, fallback=None)[0], runs=args.runs)


if __name__ == "__main__":
    main()
