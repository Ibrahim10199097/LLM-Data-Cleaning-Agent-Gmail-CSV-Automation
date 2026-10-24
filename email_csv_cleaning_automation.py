"""
Email CSV Cleaning Automation
Gmail -> download CSV attachment -> clean with Pandas -> reply with cleaned CSV

Requirements:
    pip install pandas google-api-python-client google-auth-httplib2 google-auth-oauthlib

Gmail setup:
1. Create a Google Cloud project.
2. Enable Gmail API.
3. Create OAuth Desktop credentials.
4. Save the downloaded JSON as:
       credentials.json
5. Run this script once interactively to authorize Gmail.

The script:
- Looks for unread Gmail messages containing CSV attachments.
- Downloads the CSV.
- Cleans the data.
- Creates <original_name>_cleaned.csv.
- Sends the cleaned CSV back to the original sender.
- Labels the message as Automation-Processed.
- Marks it as read.
"""

import base64
import io
import logging
import os
import re
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import pandas as pd

from cleaning_agent import clean_csv_with_agent  # LLM tool-calling agent

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build


# ============================================================
# CONFIGURATION
# ============================================================

SCOPES = [
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/gmail.send",
]

CREDENTIALS_FILE = "credentials.json"
TOKEN_FILE = "token.json"

PROCESSED_LABEL = "Automation-Processed"
FAILED_LABEL = "Automation-Failed"

MAX_FILE_SIZE_MB = 20

# Change these if you want stricter cleaning.
MISSING_TOKENS = {
    "",
    "na",
    "n/a",
    "null",
    "none",
    "-",
    "?",
    "blank",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


# ============================================================
# GMAIL AUTHENTICATION
# ============================================================

def get_gmail_service():
    creds = None

    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                CREDENTIALS_FILE,
                SCOPES,
            )
            creds = flow.run_local_server(port=0)

        with open(TOKEN_FILE, "w", encoding="utf-8") as token:
            token.write(creds.to_json())

    return build("gmail", "v1", credentials=creds)


# ============================================================
# GMAIL HELPERS
# ============================================================

def get_header(headers, name):
    for header in headers:
        if header["name"].lower() == name.lower():
            return header["value"]
    return ""


def get_or_create_label(service, label_name):
    labels = service.users().labels().list(userId="me").execute().get(
        "labels", []
    )

    for label in labels:
        if label["name"] == label_name:
            return label["id"]

    body = {
        "name": label_name,
        "labelListVisibility": "labelShow",
        "messageListVisibility": "show",
    }

    created = service.users().labels().create(
        userId="me",
        body=body,
    ).execute()

    return created["id"]


def search_messages(service):
    """
    Find unread emails with CSV attachments that have not already
    been processed.
    """

    query = (
        "is:unread "
        "has:attachment "
        "filename:csv "
        f"-label:{PROCESSED_LABEL}"
    )

    response = service.users().messages().list(
        userId="me",
        q=query,
    ).execute()

    return response.get("messages", [])


def get_message(service, message_id):
    return service.users().messages().get(
        userId="me",
        id=message_id,
        format="full",
    ).execute()


def find_csv_attachments(payload):
    """
    Recursively find CSV attachments in a Gmail message.
    """

    attachments = []

    def walk(part):
        filename = part.get("filename", "")
        body = part.get("body", {})

        if filename.lower().endswith(".csv"):
            attachments.append({
                "filename": filename,
                "attachment_id": body.get("attachmentId"),
                "size": body.get("size", 0),
                "part": part,
            })

        for child in part.get("parts", []):
            walk(child)

    walk(payload)
    return attachments


def download_attachment(service, message_id, attachment_id):
    data = service.users().messages().attachments().get(
        userId="me",
        messageId=message_id,
        id=attachment_id,
    ).execute()

    return base64.urlsafe_b64decode(data["data"])


# ============================================================
# CSV CLEANING
# ============================================================

def normalize_column_name(column):
    """
    Example:
        ' Order Amount ($) ' -> 'order_amount'
        'Customer ID'        -> 'customer_id'
    """

    column = str(column).strip().lower()

    column = re.sub(r"[^a-zA-Z0-9]+", "_", column)
    column = re.sub(r"_+", "_", column)
    column = column.strip("_")

    return column


def clean_text(value):
    if pd.isna(value):
        return pd.NA

    value = str(value)

    # Collapse repeated whitespace.
    value = re.sub(r"\s+", " ", value).strip()

    if value.lower() in MISSING_TOKENS:
        return pd.NA

    return value


def looks_like_identifier(column_name):
    """
    Columns such as phone, ZIP, SKU, IDs etc. should keep
    leading zeros and should not automatically become numeric.
    """

    identifier_words = [
        "id",
        "phone",
        "mobile",
        "zip",
        "postal",
        "postcode",
        "sku",
        "code",
    ]

    name = column_name.lower()

    return any(word in name for word in identifier_words)


def try_convert_numeric(series, column_name):
    if looks_like_identifier(column_name):
        return series

    # Remove common numeric formatting.
    cleaned = (
        series.astype("string")
        .str.replace(",", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.replace("€", "", regex=False)
        .str.replace("£", "", regex=False)
        .str.strip()
    )

    converted = pd.to_numeric(cleaned, errors="coerce")

    non_empty = series.notna().sum()

    if non_empty == 0:
        return series

    success_rate = converted.notna().sum() / non_empty

    # Only convert if at least 90% of non-empty values are numeric.
    if success_rate >= 0.90:
        return converted

    return series


def try_convert_date(series):
    converted = pd.to_datetime(
        series,
        errors="coerce",
        dayfirst=False,
    )

    non_empty = series.notna().sum()

    if non_empty == 0:
        return series

    success_rate = converted.notna().sum() / non_empty

    if success_rate >= 0.90:
        return converted.dt.strftime("%Y-%m-%d")

    return series


def clean_csv(csv_bytes):
    """
    Returns:
        cleaned_csv_bytes,
        quality_report
    """

    # --------------------------------------------------------
    # Load CSV as text first.
    # --------------------------------------------------------

    original = pd.read_csv(
        io.BytesIO(csv_bytes),
        dtype=str,
        keep_default_na=False,
    )

    rows_before = len(original)
    columns_before = len(original.columns)

    df = original.copy()

    # --------------------------------------------------------
    # Normalize column names.
    # --------------------------------------------------------

    df.columns = [
        normalize_column_name(column)
        for column in df.columns
    ]

    # --------------------------------------------------------
    # Clean cell values.
    # --------------------------------------------------------

    for column in df.columns:
        df[column] = df[column].apply(clean_text)

    # --------------------------------------------------------
    # Remove completely empty rows and columns.
    # --------------------------------------------------------

    empty_rows = int(df.isna().all(axis=1).sum())

    df = df.dropna(how="all")

    empty_columns = int(df.isna().all(axis=0).sum())

    df = df.dropna(axis=1, how="all")

    # --------------------------------------------------------
    # Remove exact duplicate rows.
    # --------------------------------------------------------

    duplicates = int(df.duplicated().sum())

    df = df.drop_duplicates()

    # --------------------------------------------------------
    # Type conversion.
    # --------------------------------------------------------

    numeric_columns = []
    date_columns = []

    for column in df.columns:

        before = df[column].copy()

        numeric_result = try_convert_numeric(
            df[column],
            column,
        )

        if not numeric_result.equals(before):
            df[column] = numeric_result
            numeric_columns.append(column)
            continue

        date_result = try_convert_date(df[column])

        if not date_result.equals(before):
            df[column] = date_result
            date_columns.append(column)

    # --------------------------------------------------------
    # Missing values.
    # --------------------------------------------------------

    missing_before = int(df.isna().sum().sum())

    for column in df.columns:

        if pd.api.types.is_numeric_dtype(df[column]):
            if df[column].notna().any():
                df[column] = df[column].fillna(
                    df[column].median()
                )

        else:
            # Date columns are allowed to remain empty.
            if column not in date_columns:
                df[column] = df[column].fillna("Unknown")

    missing_after = int(df.isna().sum().sum())

    # --------------------------------------------------------
    # Quality report.
    # --------------------------------------------------------

    report = {
        "rows_before": rows_before,
        "rows_after": len(df),
        "columns_before": columns_before,
        "columns_after": len(df.columns),
        "empty_rows_removed": empty_rows,
        "empty_columns_removed": empty_columns,
        "duplicates_removed": duplicates,
        "missing_cells_before": missing_before,
        "missing_cells_after": missing_after,
        "numeric_columns": numeric_columns,
        "date_columns": date_columns,
    }

    # --------------------------------------------------------
    # Export cleaned CSV.
    # --------------------------------------------------------

    output = io.StringIO()

    df.to_csv(
        output,
        index=False,
    )

    cleaned_bytes = output.getvalue().encode("utf-8")

    return cleaned_bytes, report


# ============================================================
# EMAIL SENDING
# ============================================================

def send_cleaned_file(
    service,
    sender_email,
    original_subject,
    cleaned_filename,
    cleaned_bytes,
    report,
):
    message = MIMEMultipart()

    message["To"] = sender_email

    if original_subject.lower().startswith("re:"):
        message["Subject"] = original_subject
    else:
        message["Subject"] = f"Re: {original_subject}"

    decisions = "\n".join(
        f"- {a['tool']} {a['args'] or ''}: {a['reason']}"
        for a in report.get("actions", [])
        if a["reason"] and not str(a["result"]).startswith("ERROR")
    )
    agent_section = (
        f"Cleaning notes ({report.get('mode', 'fixed')} mode):\n"
        f"{report.get('agent_summary', '')}\n\n"
        f"Steps taken:\n{decisions or '- standard rules'}\n"
    )

    body = f"""
Hello,

Your CSV file has been cleaned successfully.

Cleaning summary:
- Rows: {report["rows_before"]} → {report["rows_after"]}
- Columns: {report["columns_before"]} → {report["columns_after"]}
- Duplicate rows removed: {report["duplicates_removed"]}
- Empty rows removed: {report["empty_rows_removed"]}
- Empty columns removed: {report["empty_columns_removed"]}
- Missing cells before cleaning: {report["missing_cells_before"]}
- Missing cells after cleaning: {report["missing_cells_after"]}

Detected numeric columns:
{", ".join(report["numeric_columns"]) or "None"}

Detected date columns:
{", ".join(report["date_columns"]) or "None"}

{agent_section}
The cleaned CSV is attached.

Best regards,
CSV Cleaning Automation
""".strip()

    message.attach(
        MIMEText(body, "plain")
    )

    attachment = MIMEApplication(
        cleaned_bytes,
        Name=cleaned_filename,
    )

    attachment["Content-Disposition"] = (
        f'attachment; filename="{cleaned_filename}"'
    )

    message.attach(attachment)

    encoded_message = base64.urlsafe_b64encode(
        message.as_bytes()
    ).decode()

    send_body = {
        "raw": encoded_message
    }

    return service.users().messages().send(
        userId="me",
        body=send_body,
    ).execute()


# ============================================================
# LABEL / MESSAGE STATE
# ============================================================

def mark_processed(service, message_id):
    label_id = get_or_create_label(
        service,
        PROCESSED_LABEL,
    )

    service.users().messages().modify(
        userId="me",
        id=message_id,
        body={
            "addLabelIds": [label_id],
            "removeLabelIds": ["UNREAD"],
        },
    ).execute()


def mark_failed(service, message_id):
    label_id = get_or_create_label(
        service,
        FAILED_LABEL,
    )

    service.users().messages().modify(
        userId="me",
        id=message_id,
        body={
            "addLabelIds": [label_id],
        },
    ).execute()


# ============================================================
# MAIN AUTOMATION
# ============================================================

def process_email(service, message_id):
    logging.info("Processing message: %s", message_id)

    message = get_message(
        service,
        message_id,
    )

    payload = message.get("payload", {})
    headers = payload.get("headers", [])

    sender = get_header(headers, "From")
    subject = get_header(headers, "Subject")

    # Extract plain email from:
    # John Doe <john@example.com>
    match = re.search(
        r"<([^>]+)>",
        sender,
    )

    sender_email = (
        match.group(1)
        if match
        else sender
    )

    attachments = find_csv_attachments(payload)

    if not attachments:
        logging.info("No CSV attachment found.")
        return

    for attachment in attachments:

        filename = attachment["filename"]
        size = attachment.get("size", 0)

        if size > MAX_FILE_SIZE_MB * 1024 * 1024:
            raise ValueError(
                f"{filename} exceeds the {MAX_FILE_SIZE_MB} MB limit."
            )

        attachment_id = attachment["attachment_id"]

        if not attachment_id:
            raise ValueError(
                f"Could not find attachment ID for {filename}"
            )

        logging.info(
            "Downloading attachment: %s",
            filename,
        )

        csv_bytes = download_attachment(
            service,
            message_id,
            attachment_id,
        )

        logging.info(
            "Cleaning CSV: %s",
            filename,
        )

        # The agent plans the cleaning steps; the fixed pipeline is the safety net.
        cleaned_bytes, report = clean_csv_with_agent(
            csv_bytes,
            fallback=clean_csv,
        )

        base_name, extension = os.path.splitext(
            filename
        )

        cleaned_filename = (
            f"{base_name}_cleaned{extension}"
        )

        logging.info(
            "Sending cleaned file to %s",
            sender_email,
        )

        send_cleaned_file(
            service=service,
            sender_email=sender_email,
            original_subject=subject,
            cleaned_filename=cleaned_filename,
            cleaned_bytes=cleaned_bytes,
            report=report,
        )

    mark_processed(
        service,
        message_id,
    )

    logging.info(
        "Finished successfully: %s",
        message_id,
    )


def run_automation():
    service = get_gmail_service()

    messages = search_messages(
        service
    )

    if not messages:
        logging.info(
            "No new CSV emails found."
        )
        return

    logging.info(
        "Found %d email(s).",
        len(messages),
    )

    for message in messages:

        message_id = message["id"]

        try:
            process_email(
                service,
                message_id,
            )

        except Exception as exc:
            logging.exception(
                "Failed to process message %s",
                message_id,
            )

            try:
                mark_failed(
                    service,
                    message_id,
                )
            except Exception:
                logging.exception(
                    "Could not mark message as failed."
                )


if __name__ == "__main__":
    run_automation()
