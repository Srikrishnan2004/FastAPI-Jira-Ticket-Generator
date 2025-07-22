import logging
import json
import os
import requests
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from requests.auth import HTTPBasicAuth
from pydantic_settings import BaseSettings
from typing import Optional
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# --- Import Google Cloud Vertex AI SDK components ---
import vertexai
from vertexai.generative_models import GenerativeModel, HarmCategory, HarmBlockThreshold

logging.basicConfig(level=logging.INFO)

app = FastAPI()

# --- ChromaDB Configuration ---
local_model_path = "./local_sentence_transformer"
embedding_fn = SentenceTransformerEmbeddingFunction(model_name=local_model_path)
client = chromadb.PersistentClient(path="./chroma_data")
collection = client.get_or_create_collection(name="jira_issues", embedding_function=embedding_fn)

# --- Google Gemini Configuration ---
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID", "cobalt-howl-465609-b8")
GCP_LOCATION = os.getenv("GCP_LOCATION", "us-central1")


# --- Jira Configuration ---
class Settings(BaseSettings):
    jira_email: str
    jira_api_token: str
    jira_base_url: str = "https://ssn-team-j7z071w8.atlassian.net"
    project_key: str = "ECSA"
    # This is the custom field ID for "Epic Link" in many Jira setups.
    # You may need to verify this in your Jira instance's custom field settings.
    epic_link_field_id: str = "customfield_10014"

    class Config:
        env_file = ".env"


settings = Settings()
auth = HTTPBasicAuth(settings.jira_email, settings.jira_api_token)


# --- App Startup Event ---
@app.on_event("startup")
async def startup_event():
    # Initialize Vertex AI for Gemini access
    if not GCP_PROJECT_ID:
        logging.error("GCP_PROJECT_ID environment variable not set. Vertex AI initialization failed.")
        return

    try:
        vertexai.init(project=GCP_PROJECT_ID, location=GCP_LOCATION)
        logging.info(f"Vertex AI initialized for project {GCP_PROJECT_ID} in region {GCP_LOCATION}")
    except Exception as e:
        logging.error(f"Failed to initialize Vertex AI: {e}. Ensure your environment is authenticated.")


# --- Pydantic Models ---
class AddDocumentRequest(BaseModel):
    document: str
    metadata: dict
    id: str


class GenerateTicketRequest(BaseModel):
    commit_message: str
    repo: str


class GenerateClientTicketRequest(BaseModel):
    request_text: str
    repo: str


# --- API Endpoints ---
@app.post("/add")
def add_document(request: AddDocumentRequest):
    collection.add(
        documents=[request.document],
        metadatas=[request.metadata],
        ids=[request.id]
    )
    return {"status": "Document added successfully"}


@app.post("/generate_ticket")
def generate_ticket(request: GenerateTicketRequest):
    logging.info("Attempting to generate ticket for: %s", request.commit_message)

    gatekeeper_system_instruction = """
    You are an expert JIRA Ticket Automation Agent. Your **SOLE PURPOSE** is to classify a given commit message as either "Substantive" or "Non-Substantive".
    **CRITICAL RULE: You MUST respond ONLY with a single JSON object.**
    **A. Non-Substantive Commits:** Respond ONLY with: `{"status": "No Jira ticket generated. The commit message was identified as a non-substantive update."}`
    **B. Substantive Commits:** Respond ONLY with: `{"status": "Jira ticket required. The commit message indicates a substantive change."}`
    """
    try:
        gatekeeper_model = GenerativeModel("gemini-2.5-pro", system_instruction=[gatekeeper_system_instruction])
        gatekeeper_response = gatekeeper_model.generate_content(
            [f"Commit Message: {request.commit_message}"],
            generation_config={"temperature": 0.1, "response_mime_type": "application/json"},
            safety_settings={category: HarmBlockThreshold.BLOCK_NONE for category in HarmCategory}
        )
        gatekeeper_parsed_json = json.loads(gatekeeper_response.text)
        if "No Jira ticket generated" in gatekeeper_parsed_json.get("status", ""):
            return gatekeeper_parsed_json
        if "Jira ticket required" not in gatekeeper_parsed_json.get("status", ""):
            raise HTTPException(status_code=500, detail="Gatekeeper LLM returned an unexpected status.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during commit classification: {e}")

    logging.info("Commit identified as substantive. Proceeding to generate Jira ticket.")

    # --- ChromaDB Search Logic ---
    search_priority = ["Epic", "Story", "Task", "Sub-task", "Bug"]
    matched_metadata, matched_document, parent_chain = None, None, []

    def find_best_descendant(parent_key, child_issue_type):
        results = collection.query(
            query_texts=[request.commit_message], n_results=1,
            where={"$and": [{"repo": request.repo}, {"issue_type": child_issue_type}, {"parent": parent_key}]},
            include=["metadatas", "documents", "distances"]
        )
        if results and results.get("ids") and results["ids"][0]:
            child_metadata = dict(results["metadatas"][0][0]);
            child_metadata["key"] = results["ids"][0][0]
            return child_metadata, results["documents"][0][0]
        return None, None

    for i, issue_type in enumerate(search_priority):
        results = collection.query(
            query_texts=[request.commit_message], n_results=1,
            where={"$and": [{"repo": request.repo}, {"issue_type": issue_type}]},
            include=["metadatas", "documents", "distances"]
        )
        if results and results.get("ids") and results["ids"][0]:
            matched_metadata = dict(results["metadatas"][0][0]);
            matched_metadata["key"] = results["ids"][0][0]
            matched_document = results["documents"][0][0]
            parent_chain.append({k: matched_metadata.get(k, "N/A") for k in ["key", "summary", "issue_type"]})
            parent_key = matched_metadata.get("key")
            for child_issue_type in search_priority[i + 1:]:
                child_meta, child_doc = find_best_descendant(parent_key, child_issue_type)
                if child_meta:
                    parent_chain.append({k: child_meta.get(k, "N/A") for k in ["key", "summary", "issue_type"]})
                    matched_metadata, matched_document, parent_key = child_meta, child_doc, child_meta.get("key")
                else:
                    break
            break

    unique_chain = []
    if parent_chain:
        seen_keys = set();
        [unique_chain.insert(0, p) for p in reversed(parent_chain) if
         p['key'] not in seen_keys and not seen_keys.add(p['key'])]
    hierarchy_view = "Hierarchy:\n" + "\n".join([f"- {p['issue_type']}: {p['key']} - {p['summary']}" for p in
                                                 unique_chain]) if unique_chain else "No hierarchy found."

    ticket_generation_system_instruction = """
You are an expert JIRA Ticket Automation Agent. Analyze a commit message and generate a JIRA ticket.
**Definitions:**
- **Task:** A technical to-do.
- **Bug:** A fix for an error.
- **Sub-task:** A small part of a larger task.
**CRITICAL RULE: Respond with a single JSON object.**
Schema: {{ "summary": "string", "description": "string", "issue_type": "Task" or "Bug" or "Sub-task", "parent": "string" or "N/A" }}
**Rules:**
1.  **Analyze:** Classify the commit as a Task, Bug, or Sub-task.
2.  **Description:** Must start with "Test" or "Verify".
3.  **Parent:** Use the key from the hierarchy context, else "N/A".
"""
    ticket_generation_user_prompt = f"Commit Message: {request.commit_message}\nRepository: {request.repo}\n\n{hierarchy_view}\n\nGenerate the JIRA ticket."

    try:
        logging.info("Calling Ticket Generation LLM (Gemini).")
        ticket_model = GenerativeModel("gemini-2.5-pro", system_instruction=[ticket_generation_system_instruction])
        response = ticket_model.generate_content(
            [ticket_generation_user_prompt],
            generation_config={"temperature": 0.5, "response_mime_type": "application/json"},
            safety_settings={category: HarmBlockThreshold.BLOCK_NONE for category in HarmCategory}
        )
        ticket_details = json.loads(response.text.strip())

        required_keys = ["summary", "description", "issue_type", "parent"]
        if not all(key in ticket_details for key in required_keys):
            raise ValueError("Generated ticket is missing one or more required keys.")
        if ticket_details["issue_type"] not in ["Task", "Bug", "Sub-task"]:
            raise ValueError(f"Invalid issue_type generated: '{ticket_details['issue_type']}'.")

        # --- Create Jira Issue ---
        url = f"{settings.jira_base_url}/rest/api/3/issue"
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        payload = {"fields": {
            "project": {"key": settings.project_key},
            "summary": ticket_details["summary"],
            "description": ticket_details["description"],
            "issuetype": {"name": ticket_details["issue_type"]}
        }}

        if ticket_details["issue_type"] == "Sub-task" and ticket_details["parent"] != "N/A":
            payload["fields"]["parent"] = {"key": ticket_details["parent"]}
        elif ticket_details["parent"] != "N/A":
            # Assume parent is an Epic for Tasks and Bugs
            payload["fields"][settings.epic_link_field_id] = ticket_details["parent"]

        jira_response = requests.post(url, headers=headers, auth=auth, json=payload)
        jira_response.raise_for_status()  # Raise an exception for bad status codes

        return {"message": "Jira ticket created successfully", "key": jira_response.json()["key"],
                "generated_details": ticket_details}

    except requests.exceptions.HTTPError as e:
        raise HTTPException(status_code=e.response.status_code,
                            detail=f"Failed to create Jira issue: {e.response.text}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")


@app.post("/generate_client_ticket")
def generate_client_ticket(request: GenerateClientTicketRequest):
    logging.info("Attempting to generate client ticket for: %s", request.request_text)

    gatekeeper_system_instruction = """
    You are a requirements analyst AI. Classify a client's request.
    **CRITICAL RULE: Respond ONLY with a single JSON object.**
    **A. Client Feature Request:** `{"status": "Actionable request received. Proceeding with ticket generation."}`
    **B. Developer Task:** `{"status": "Ticket not generated. This appears to be a developer task, not a client feature request."}`
    **C. Vague Request:** `{"status": "Ticket not generated. The request is too vague. Please provide more details."}`
    """
    try:
        gatekeeper_model = GenerativeModel("gemini-2.5-pro", system_instruction=[gatekeeper_system_instruction])
        gatekeeper_response = gatekeeper_model.generate_content(
            [f"Client Request: {request.request_text}"],
            generation_config={"temperature": 0.1, "response_mime_type": "application/json"},
            safety_settings={category: HarmBlockThreshold.BLOCK_NONE for category in HarmCategory}
        )
        gatekeeper_parsed_json = json.loads(gatekeeper_response.text.strip())
        if "Ticket not generated" in gatekeeper_parsed_json.get("status", ""):
            return gatekeeper_parsed_json
        if "Actionable request received" not in gatekeeper_parsed_json.get("status", ""):
            raise HTTPException(status_code=500, detail="Gatekeeper LLM returned an unexpected status.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during request classification: {e}")

    logging.info("Request is actionable. Searching for related Epics.")
    epic_search_results = collection.query(
        query_texts=[request.request_text], n_results=1,
        where={"$and": [{"repo": request.repo}, {"issue_type": "Epic"}]},
        include=["metadatas","documents", "distances"]
    )
    related_epic_view, parent_epic_key = "No related Epic found.", "N/A"
    if epic_search_results and epic_search_results.get("ids") and epic_search_results["ids"][0]:
        epic_meta = epic_search_results["metadatas"][0][0]
        parent_epic_key = epic_search_results["ids"][0][0]
        related_epic_view = f"Found a potentially related Epic:\n- {parent_epic_key}: {epic_meta.get('summary')}"

    ticket_generation_system_instruction = """
You are an expert JIRA Product Manager AI. Convert a client's request into a JIRA Epic or Story.
**Definitions:**
- **Epic:** A large feature grouping multiple stories (e.g., "User Profile System").
- **Story:** A specific user request describing value (e.g., "As a user, I want to upload a picture...").
**CRITICAL RULE: Respond with a single JSON object.**
Schema: {{ "summary": "string", "description": "string", "issue_type": "Epic" or "Story", "parent": "string" or "N/A" }}
**Rules:**
1.  **Analyze:** Classify the request as a broad Epic or a specific Story.
2.  **Epic Summary:** If an Epic, the summary must be a short, high-level title (2-4 words).
3.  **Story Description:** If a Story, the description MUST be in user story format.
4.  **User Type:** The user type MUST be one of: 'user', 'client', 'mobile user', or 'desktop user'.
5.  **Parent Logic:** An Epic's parent is "N/A". A Story's parent should be the related Epic key.
"""
    ticket_generation_user_prompt = f"Client's Request: \"{request.request_text}\"\n\nContext:\n{related_epic_view}\n\nGenerate the JIRA ticket as a JSON object."
    try:
        logging.info("Calling Ticket Generation LLM (Gemini) for client ticket.")
        ticket_model = GenerativeModel("gemini-2.5-pro", system_instruction=[ticket_generation_system_instruction])
        response = ticket_model.generate_content(
            [ticket_generation_user_prompt],
            generation_config={"temperature": 0.6, "response_mime_type": "application/json"},
            safety_settings={category: HarmBlockThreshold.BLOCK_NONE for category in HarmCategory}
        )
        ticket_details = json.loads(response.text.strip())

        required_keys = ["summary", "description", "issue_type", "parent"]
        if not all(key in ticket_details and ticket_details[key] is not None for key in required_keys):
            raise ValueError("Generated ticket is missing required keys or they are null.")
        if ticket_details["issue_type"] not in ["Epic", "Story"]:
            raise ValueError(f"Invalid issue_type generated: '{ticket_details['issue_type']}'.")

        # --- Create Jira Issue ---
        url = f"{settings.jira_base_url}/rest/api/3/issue"
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        payload = {"fields": {
            "project": {"key": settings.project_key},
            "summary": ticket_details["summary"],
            "description": ticket_details["description"],
            "issuetype": {"name": ticket_details["issue_type"]}
        }}

        if ticket_details["issue_type"] == "Story" and parent_epic_key != "N/A":
            payload["fields"][settings.epic_link_field_id] = parent_epic_key

        jira_response = requests.post(url, headers=headers, auth=auth, json=payload)
        jira_response.raise_for_status()

        return {"message": "Jira ticket created successfully", "key": jira_response.json()["key"],
                "generated_details": ticket_details}
    except requests.exceptions.HTTPError as e:
        raise HTTPException(status_code=e.response.status_code,
                            detail=f"Failed to create Jira issue: {e.response.text}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")


@app.post("/clear_collection")
def clear_collection():
    try:
        num_items = collection.count()
        if num_items == 0:
            return {"status": "Collection was already empty."}
        all_items = collection.get(limit=num_items)
        all_ids = all_items.get("ids")
        if all_ids:
            collection.delete(ids=all_ids)
            logging.info(f"Cleared {len(all_ids)} documents from collection.")
        return {"status": "Collection cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
