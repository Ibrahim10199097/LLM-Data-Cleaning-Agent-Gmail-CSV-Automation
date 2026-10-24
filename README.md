# CSV Cleaning Agent

Send a messy CSV to Gmail. An AI agent cleans it and replies with the cleaned file
and a short explanation of what it did.

## What it does

1. Finds unread emails with a CSV attachment.
2. Looks at the data (columns, missing values, sample rows).
3. An LLM agent (LangGraph) chooses the cleaning steps, for example:
   remove duplicates, fix dates, convert numbers, fill missing values, merge category spellings.
4. Replies to the sender with the cleaned CSV and the list of steps.

If the agent fails, the original rule-based cleaning runs instead, so no email is lost.

## Why an agent?

The old version used the same fixed steps for every file. It broke on things like
phone numbers with leading zeros, `dd/mm/yyyy` dates and prices like `EGP 950`.
The agent looks at each file and decides what to do. Each tool also checks itself and
refuses unsafe actions (for example, it will not turn `00123` into `123`).

## Results

Tested on 8 small messy CSV files with known correct answers (`eval_agent.py`).

| Version | Files fully correct |
|---|---|
| Fixed rules (original) | 3 / 8 |
| LLM agent | _X / 8_ |

## How to run

```bash
pip install -r requirements.txt
```

1. **Gmail:** enable the Gmail API in Google Cloud, create OAuth Desktop credentials,
   and save the file as `credentials.json` in this folder.
2. **LLM key:** get a free Gemini key and set it in the terminal:
   ```powershell
   $env:GOOGLE_API_KEY="your-key"
   ```
3. **Test it:**
   ```bash
   python eval_agent.py --agent
   ```
4. **Run it:**
   ```bash
   python email_csv_cleaning_automation.py
   ```
   The first run opens a browser to log in to Gmail. After that, email a CSV to that
   account and run the script again.

## Files

| File | Purpose |
|---|---|
| `cleaning_agent.py` | The agent and its cleaning tools |
| `email_csv_cleaning_automation.py` | Gmail part + the original fixed rules |
| `eval_agent.py` | Test files that compare fixed rules and the agent |

## Notes

- Only `.csv` files for now.
- Never upload `credentials.json` or `token.json` (they are in `.gitignore`).

**Tech:** Python, Pandas, LangGraph, LangChain, Gmail API
