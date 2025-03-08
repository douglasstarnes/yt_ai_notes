import datetime
import re

from fastapi import FastAPI
from markdown_pdf import MarkdownPdf, Section
from ollama import generate, GenerateResponse
from pydantic import BaseModel

app = FastAPI()

class MeetingNotes(BaseModel):
    subject: str
    attendees: list[str]
    transcript: str
    date: datetime.date

def to_snake_case(text):
    text = re.sub(r"[^a-zA-Z0-9\s-]", "", text)
    text = re.sub(r"[\s-]+", "_", text)

    return text.lower()

@app.post("/summarize_meeting")
def summarize_meeting(meeting_notes: MeetingNotes):
    prompt = f"""
You are an AI office assistant.

Analyze the meeting notes for the following:
1. A brief summary of the transcript
2. Extract the main topics
3. Any tasks, the attendees they are assigned to, and the due date.

Subject: {meeting_notes.subject}
Date: {meeting_notes.date.strftime("%Y/%m/%d")}
Attendees: {",".join(meeting_notes.attendees)}
Transcript: {meeting_notes.transcript}

Format the summary as follows:

SUBJECT: 
[meeting subject]

DATE: 
[meeting date in MM/DD/YYYY format]

SUMMARY: 
[meeting summary]

TOPICS: 
[list of main topics]

TASKS: 
[list of tasks, the person assigned to the task and the due date in MM/DD/YYYY format]

FOLLOW UP: 
[draft a follow up email for one week after meeting date]
"""
    response: GenerateResponse = generate(
        model="llama3.1:8b",
        prompt=prompt,
        stream=False
    )

    filename = f"{to_snake_case(meeting_notes.subject)}.pdf"

    pdf = MarkdownPdf()
    pdf.add_section(Section(response.response))
    pdf.save(filename)

    return {"filename": filename}