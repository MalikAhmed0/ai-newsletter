#!/usr/bin/env python3
"""
The AI Pulse — Personalized Daily Newsletter
=============================================
• Uses Serper.dev for Google Search (free tier: 2,500 searches)
• Uses Gemini 2.0 Flash free tier for writing (no grounding needed)
• Reads recipients from recipients.json
• Loads per-user preferences from preferences/{uid}.json
• Sends personalized HTML email via Gmail SMTP
"""

import hashlib
import json
import logging
import os
import smtplib
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from urllib.parse import quote
from urllib.request import urlopen

import pytz
import requests
from google import genai
from google.genai import types

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
PKT = pytz.timezone("Asia/Karachi")

GITHUB_REPOSITORY = os.environ.get("GITHUB_REPOSITORY", "")
GITHUB_USERNAME   = GITHUB_REPOSITORY.split("/")[0] if "/" in GITHUB_REPOSITORY else ""
REPO_NAME         = GITHUB_REPOSITORY.split("/")[1] if "/" in GITHUB_REPOSITORY else "ai-newsletter"
FEEDBACK_BASE_URL = f"https://{GITHUB_USERNAME}.github.io/{REPO_NAME}/feedback.html"

SEARCH_QUERIES = [
    "LLM GPT Claude Gemini new model release today",
    "AI tools apps product launch this week",
    "artificial intelligence business enterprise deployment news",
    "AI research breakthrough paper published today",
    "AI regulation policy safety government news",
    "wild funny surprising AI story today",
]


# ── Recipients & Preferences ──────────────────────────────────────────────────
def load_recipients() -> list:
    with open("recipients.json") as f:
        return json.load(f)


def get_uid(email: str) -> str:
    return hashlib.md5(email.lower().strip().encode()).hexdigest()


def load_preferences(uid: str) -> dict | None:
    url = (
        f"https://raw.githubusercontent.com/{GITHUB_REPOSITORY}"
        f"/main/preferences/{uid}.json"
    )
    try:
        with urlopen(url, timeout=8) as r:
            return json.loads(r.read())
    except Exception:
        return None


def prefs_context(prefs: dict | None) -> str:
    if not prefs:
        return ""
    lines = []
    history = prefs.get("rating_history", [])
    if history:
        avg = sum(history) / len(history)
        if avg < 3:
            lines.append(f"- Rating average {avg:.1f}/5 — reader is NOT satisfied. Adjust significantly.")
        elif avg < 4:
            lines.append(f"- Rating average {avg:.1f}/5 — some room to improve.")
        else:
            lines.append(f"- Rating average {avg:.1f}/5 — reader enjoys this newsletter.")
    if prefs.get("instructions"):
        lines.append(f"- Reader's instructions: \"{prefs['instructions']}\"")
    if prefs.get("detail_level"):
        lines.append(f"- Preferred detail level: {prefs['detail_level']}")
    if prefs.get("tone_preference"):
        lines.append(f"- Preferred tone: {prefs['tone_preference']}")
    if prefs.get("avoid"):
        lines.append(f"- They want LESS of: {prefs['avoid']}")
    if prefs.get("more_of"):
        lines.append(f"- They want MORE of: {prefs['more_of']}")
    if not lines:
        return ""
    return (
        "\n\n━━━ PERSONALIZATION FOR THIS READER ━━━\n"
        + "\n".join(lines)
        + "\nApply these preferences when selecting and writing stories."
    )


# ── Serper Search ─────────────────────────────────────────────────────────────
def search_news(queries: list) -> str:
    api_key = os.environ["SERPER_API_KEY"]
    all_results = []

    for query in queries:
        try:
            resp = requests.post(
                "https://google.serper.dev/news",
                headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
                json={"q": query, "num": 5, "tbs": "qdr:d2"},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            for item in data.get("news", []):
                all_results.append(
                    f"- {item.get('title','')}\n"
                    f"  Source: {item.get('source','')}\n"
                    f"  Date: {item.get('date','')}\n"
                    f"  Snippet: {item.get('snippet','')}\n"
                    f"  URL: {item.get('link','')}"
                )
        except Exception as e:
            log.warning(f"Serper search failed for '{query}': {e}")

    log.info(f"Fetched {len(all_results)} news items from Serper.")
    return "\n\n".join(all_results)


# ── Content Generation ────────────────────────────────────────────────────────
def generate_content(client: genai.Client, date_str: str, prefs: dict | None) -> dict:
    news_raw = search_news(SEARCH_QUERIES)
    extra    = prefs_context(prefs)

    prompt = f"""Today is {date_str}. You are an expert AI journalist writing "The AI Pulse," a daily newsletter.

Below is raw news data pulled from the web in the last 48 hours. Use it as your only source material.

━━━ RAW NEWS DATA ━━━
{news_raw}
━━━ END OF NEWS DATA ━━━
{extra}

Using ONLY the news above (do not invent stories), write today's newsletter.
Return ONLY a valid JSON object — no markdown fences, no preamble. Schema:

{{
  "top_story": {{
    "headline": "Punchy, specific headline naming the company/model/person",
    "summary": "2–3 sentences. What happened and why it changed things.",
    "why_it_matters": "One sentence that makes the reader feel smart for knowing this.",
    "source": "Publication name",
    "url": "Full URL from the data above or empty string"
  }},
  "stories": [
    {{
      "topic": "One of: Tools | Business | Research | Policy | LLMs",
      "headline": "Specific headline",
      "summary": "2–3 concrete sentences. No vague hype.",
      "why_it_matters": "One insight sentence.",
      "source": "Publication name",
      "url": "Full URL from the data above or empty string"
    }}
  ],
  "wild_story": {{
    "headline": "Something surprising, funny, or jaw-dropping from the data",
    "summary": "2–3 sentences. Quirky, absurd, or mind-bending.",
    "source": "Publication name",
    "url": "Full URL from the data above or empty string"
  }},
  "quick_bites": [
    "Punchy one-liner — a fact or stat worth repeating",
    "Punchy one-liner #2",
    "Punchy one-liner #3"
  ]
}}

Rules:
- stories must have exactly 5 items covering different topic areas
- Pick the most impressive, credible, recent stories from the data
- Headlines must name specific companies, models, or people
- wild_story must be genuinely surprising — not another serious piece
- quick_bites should be punchy facts someone would repeat at dinner
- Only use URLs that appear in the raw data above
"""

    log.info("Calling Gemini to write newsletter…")
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.65),
    )

    raw = response.text.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) >= 2 else raw
        if raw.lstrip().startswith("json"):
            raw = raw.lstrip()[4:]
    raw = raw.strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        log.error(f"JSON parse failed: {e}\nRaw:\n{raw[:600]}")
        raise


# ── HTML Rendering ────────────────────────────────────────────────────────────
TOPIC_COLORS = {
    "llm": "#6366f1", "model": "#6366f1", "frontier": "#6366f1",
    "tool": "#10b981", "app": "#10b981",
    "business": "#f59e0b", "enterprise": "#f59e0b",
    "research": "#3b82f6", "paper": "#3b82f6",
    "policy": "#ef4444", "regulation": "#ef4444", "safety": "#ef4444",
}

def topic_color(label: str) -> str:
    lower = label.lower()
    for k, v in TOPIC_COLORS.items():
        if k in lower:
            return v
    return "#8b5cf6"


def render_html(data: dict, date_str: str, recipient_name: str, feedback_url: str) -> str:
    top     = data["top_story"]
    stories = data["stories"]
    wild    = data["wild_story"]
    bites   = data["quick_bites"]

    def read_more(url: str, color: str = "#6366f1") -> str:
        if url and url.startswith("http"):
            return (
                f'<a href="{url}" style="color:{color};font-size:12px;'
                f'text-decoration:none;font-weight:600;">Read More →</a>'
            )
        return ""

    stories_html = ""
    for s in stories:
        c = topic_color(s.get("topic", ""))
        stories_html += f"""
        <div style="border-left:4px solid {c};padding:18px 20px;margin-bottom:22px;
                    background:#f8f9ff;border-radius:0 10px 10px 0;">
          <div style="font-size:10px;font-weight:800;letter-spacing:1.5px;
                      text-transform:uppercase;color:{c};margin-bottom:8px;">
            {s.get("topic","News")}
          </div>
          <div style="font-size:16px;font-weight:700;color:#0f172a;
                      margin-bottom:8px;line-height:1.4;">
            {s["headline"]}
          </div>
          <div style="font-size:14px;color:#475569;line-height:1.65;margin-bottom:10px;">
            {s["summary"]}
          </div>
          <div style="font-size:13px;color:#64748b;font-style:italic;
                      background:#eef2ff;padding:8px 12px;border-radius:6px;
                      margin-bottom:10px;">
            💡 {s["why_it_matters"]}
          </div>
          <div style="display:flex;justify-content:space-between;align-items:center;">
            <span style="font-size:11px;color:#94a3b8;">— {s.get("source","")}</span>
            {read_more(s.get("url",""), c)}
          </div>
        </div>"""

    bites_html = "".join(
        f'<li style="margin-bottom:12px;font-size:14px;color:#e2e8f0;'
        f'line-height:1.5;">{b}</li>'
        for b in bites
    )

    stars_html = "".join(
        f'<a href="{feedback_url}&rating={i}" '
        f'style="font-size:30px;text-decoration:none;margin:0 3px;color:#f59e0b;">★</a>'
        for i in range(1, 6)
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>The AI Pulse — {date_str}</title>
</head>
<body style="margin:0;padding:0;background:#eef0f4;
             font-family:'Segoe UI',Helvetica,Arial,sans-serif;">
<div style="max-width:640px;margin:0 auto;padding:24px 12px 40px;">

  <!-- HEADER -->
  <div style="background:linear-gradient(145deg,#0f0c29,#302b63,#24243e);
              border-radius:18px;padding:40px 32px 32px;text-align:center;
              margin-bottom:20px;">
    <div style="font-size:10px;letter-spacing:4px;text-transform:uppercase;
                color:#a78bfa;margin-bottom:10px;font-weight:700;">
      ◈ Daily Intelligence Brief ◈
    </div>
    <div style="font-size:42px;font-weight:900;color:#ffffff;
                letter-spacing:-1.5px;line-height:1;">
      The AI Pulse
    </div>
    <div style="width:60px;height:3px;background:linear-gradient(90deg,#6366f1,#a78bfa);
                margin:16px auto;border-radius:2px;"></div>
    <div style="font-size:13px;color:#94a3b8;">{date_str}</div>
    <div style="font-size:13px;color:#c4b5fd;margin-top:6px;">
      Curated for {recipient_name}
    </div>
  </div>

  <!-- TOP STORY -->
  <div style="background:linear-gradient(135deg,#4f46e5,#7c3aed);
              border-radius:14px;padding:30px;margin-bottom:20px;color:#fff;">
    <div style="font-size:10px;letter-spacing:2.5px;text-transform:uppercase;
                opacity:0.75;margin-bottom:12px;font-weight:700;">⚡ Top Story</div>
    <div style="font-size:22px;font-weight:800;line-height:1.35;margin-bottom:14px;">
      {top["headline"]}
    </div>
    <div style="font-size:14px;opacity:0.92;line-height:1.7;margin-bottom:14px;">
      {top["summary"]}
    </div>
    <div style="background:rgba(255,255,255,0.15);border-radius:8px;
                padding:12px 16px;margin-bottom:14px;font-size:13px;line-height:1.5;">
      💡 <strong>Why it matters:</strong> {top["why_it_matters"]}
    </div>
    <div style="display:flex;justify-content:space-between;align-items:center;">
      <span style="font-size:12px;opacity:0.6;">— {top.get("source","")}</span>
      {read_more(top.get("url",""), "#fff")}
    </div>
  </div>

  <!-- MAIN STORIES -->
  <div style="background:#ffffff;border-radius:14px;padding:28px;
              margin-bottom:20px;box-shadow:0 2px 12px rgba(0,0,0,0.06);">
    <div style="font-size:10px;letter-spacing:2.5px;text-transform:uppercase;
                color:#6366f1;margin-bottom:22px;font-weight:800;">
      📰 Today's Developments
    </div>
    {stories_html}
  </div>

  <!-- WILD CARD -->
  <div style="background:#ffffff;border:2px solid #ede9fe;border-radius:14px;
              padding:26px;margin-bottom:20px;">
    <div style="font-size:10px;letter-spacing:2.5px;text-transform:uppercase;
                color:#8b5cf6;margin-bottom:12px;font-weight:800;">🤯 Wild Card</div>
    <div style="font-size:19px;font-weight:700;color:#0f172a;
                margin-bottom:10px;line-height:1.4;">
      {wild["headline"]}
    </div>
    <div style="font-size:14px;color:#475569;line-height:1.65;margin-bottom:12px;">
      {wild["summary"]}
    </div>
    <div style="display:flex;justify-content:space-between;align-items:center;">
      <span style="font-size:12px;color:#94a3b8;">— {wild.get("source","")}</span>
      {read_more(wild.get("url",""))}
    </div>
  </div>

  <!-- QUICK BITES -->
  <div style="background:linear-gradient(145deg,#0f0c29,#1e1b4b);
              border-radius:14px;padding:28px;margin-bottom:20px;">
    <div style="font-size:10px;letter-spacing:2.5px;text-transform:uppercase;
                color:#a78bfa;margin-bottom:18px;font-weight:800;">⚡ Quick Bites</div>
    <ul style="margin:0;padding-left:18px;">{bites_html}</ul>
  </div>

  <!-- FEEDBACK -->
  <div style="background:#ffffff;border:2px solid #f1f5f9;border-radius:14px;
              padding:28px;margin-bottom:20px;text-align:center;">
    <div style="font-size:11px;letter-spacing:2px;text-transform:uppercase;
                color:#94a3b8;margin-bottom:12px;font-weight:700;">
      How was today's edition, {recipient_name}?
    </div>
    <div style="margin-bottom:18px;">{stars_html}</div>
    <a href="{feedback_url}"
       style="display:inline-block;background:linear-gradient(135deg,#6366f1,#8b5cf6);
              color:#fff;text-decoration:none;padding:13px 30px;border-radius:8px;
              font-size:14px;font-weight:700;">
      ✏️ Rate &amp; Customize My Newsletter
    </a>
    <div style="font-size:11px;color:#94a3b8;margin-top:12px;line-height:1.7;">
      Adjust topics, depth, and tone of <em>your</em> newsletter.<br>
      Your feedback only affects your edition — not anyone else's.
    </div>
  </div>

  <!-- FOOTER -->
  <div style="text-align:center;padding:12px;color:#94a3b8;font-size:11px;line-height:1.8;">
    <div>The AI Pulse · Personalized daily digest</div>
    <div>{date_str} · Powered by Serper + Gemini 2.0 Flash</div>
  </div>

</div>
</body>
</html>"""


# ── Email Sending ─────────────────────────────────────────────────────────────
def send_email(html: str, subject: str, to_email: str, to_name: str) -> None:
    sender   = os.environ["GMAIL_ADDRESS"]
    password = os.environ["GMAIL_APP_PASSWORD"]

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = f"The AI Pulse <{sender}>"
    msg["To"]      = f"{to_name} <{to_email}>"
    msg.attach(MIMEText(html, "html"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(sender, password)
        server.sendmail(sender, to_email, msg.as_string())

    log.info(f"✅  Sent to {to_email}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    log.info("━━━ The AI Pulse — starting ━━━")

    required = ["GEMINI_API_KEY", "SERPER_API_KEY", "GMAIL_ADDRESS", "GMAIL_APP_PASSWORD"]
    missing  = [v for v in required if not os.environ.get(v)]
    if missing:
        raise EnvironmentError(f"Missing environment variables: {missing}")

    now      = datetime.now(PKT)
    date_str = now.strftime("%A, %B %d, %Y")
    subject  = f"⚡ The AI Pulse — {now.strftime('%b %d, %Y')}"

    client     = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    recipients = load_recipients()
    log.info(f"Sending to {len(recipients)} recipient(s)…")

    for r in recipients:
        email = r["email"]
        name  = r.get("name") or email.split("@")[0].capitalize()
        uid   = get_uid(email)

        log.info(f"Processing {email}…")
        prefs = load_preferences(uid)
        log.info(f"  Preferences: {'loaded' if prefs else 'none yet (using defaults)'}")

        feedback_url = f"{FEEDBACK_BASE_URL}?email={quote(email)}&uid={uid}"
        data         = generate_content(client, date_str, prefs)
        log.info(f"  Top story: {data['top_story']['headline'][:60]}…")

        html = render_html(data, date_str, name, feedback_url)
        send_email(html, subject, email, name)

    log.info(f"━━━ Done — {len(recipients)} newsletter(s) sent ━━━")


if __name__ == "__main__":
    main()
