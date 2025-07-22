import logging
import json
import os
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
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

JIRA_ISSUES_TABLE_CONTENT = ""
JIRA_ISSUES_TABLE_PATH = "jira_issues_table.json"

# --- Google Gemini Configuration ---
# IMPORTANT: Ensure these environment variables are set in your execution environment.
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID","cobalt-howl-465609-b8")
GCP_LOCATION = os.getenv("GCP_LOCATION", "us-central1")

# --- App Startup Event ---
@app.on_event("startup")
async def startup_event():
    global JIRA_ISSUES_TABLE_CONTENT
    if os.path.exists(JIRA_ISSUES_TABLE_PATH):
        try:
            with open(JIRA_ISSUES_TABLE_PATH, 'r', encoding='utf-8') as f:
                JIRA_ISSUES_TABLE_CONTENT = f.read()
            logging.info(f"Successfully loaded {JIRA_ISSUES_TABLE_PATH}")
        except Exception as e:
            logging.error(f"Failed to load {JIRA_ISSUES_TABLE_PATH}: {e}")
            JIRA_ISSUES_TABLE_CONTENT = ""
    else:
        logging.warning(f"{JIRA_ISSUES_TABLE_PATH} not found.")

    # Initialize Vertex AI for Gemini access
    if not GCP_PROJECT_ID:
        logging.error("GCP_PROJECT_ID environment variable not set. Vertex AI initialization failed.")
        return

    try:
        vertexai.init(project=GCP_PROJECT_ID, location=GCP_LOCATION)
        logging.info(f"Vertex AI initialized for project {GCP_PROJECT_ID} in region {GCP_LOCATION}")
    except Exception as e:
        logging.error(f"Failed to initialize Vertex AI: {e}. Ensure your environment is authenticated (e.g., via 'gcloud auth application-default login').")

# --- Pydantic Models ---
class AddDocumentRequest(BaseModel):
    document: str
    metadata: dict
    id: str

class SearchRequest(BaseModel):
    query: str
    repo: Optional[str] = None

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

@app.post("/search")
def search_documents(query: str = Query(...), repo: Optional[str] = Query(None)):
    if repo:
        results = collection.query(
            query_texts=[query], n_results=5, where={"repo": repo},
            include=["metadatas", "documents", "distances"]
        )
    else:
        results = collection.query(
            query_texts=[query], n_results=5,
            include=["metadatas", "documents", "distances"]
        )
    return results

@app.post("/generate_ticket")
def generate_ticket(request: GenerateTicketRequest):
    logging.info("Attempting to generate ticket for: %s", request.commit_message)

    gatekeeper_system_instruction = """
    You are an expert JIRA Ticket Automation Agent. Your **SOLE PURPOSE** is to classify a given commit message as either "Substantive" or "Non-Substantive".
    **CRITICAL RULE: You MUST respond ONLY with a single JSON object. No other text or explanation.**
    **Classification Rules:**
    **A. Non-Substantive Commits:**
    These are routine maintenance, documentation, or trivial changes. If a commit clearly falls into *any* of these categories, you **MUST respond ONLY** with the following exact JSON object:
    `{"status": "No Jira ticket generated. The commit message was identified as a non-substantive update."}`
    **B. Substantive Commits:**
    These are changes that directly impact the application's functionality. If a commit falls into any of these categories, you **MUST respond ONLY** with the following exact JSON object:
    `{"status": "Jira ticket required. The commit message indicates a substantive change."}`
    """
    gatekeeper_user_prompt = f"Commit Message: {request.commit_message}"

    try:
        logging.info("Calling Gatekeeper LLM (Gemini) for classification.")
        gatekeeper_model = GenerativeModel(
            "gemini-2.5-pro",
            system_instruction=[gatekeeper_system_instruction]
        )
        gatekeeper_response = gatekeeper_model.generate_content(
            [gatekeeper_user_prompt],
            generation_config={"temperature": 0.1, "response_mime_type": "application/json"},
            safety_settings={
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
        )
        gatekeeper_response_text = gatekeeper_response.text.strip()
        logging.info(f"Gatekeeper LLM raw response: {gatekeeper_response_text}")
        gatekeeper_parsed_json = json.loads(gatekeeper_response_text)

        if "No Jira ticket generated" in gatekeeper_parsed_json.get("status", ""):
            return gatekeeper_parsed_json
        if "Jira ticket required" not in gatekeeper_parsed_json.get("status", ""):
            raise HTTPException(status_code=500, detail="Gatekeeper LLM returned an unexpected status.")

    except Exception as e:
        logging.error(f"Error during Gatekeeper LLM call: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred during commit classification: {e}")

    logging.info("Commit identified as substantive. Proceeding to generate Jira ticket.")
    
    search_priority = ["Epic", "Story", "Task", "Sub-task", "Bug"]
    matched_metadata = None
    matched_document = None
    parent_chain = []
    unique_chain = []

    def find_best_descendant(parent_key, child_issue_type):
        results = collection.query(
            query_texts=[request.commit_message], n_results=1,
            where={"$and": [{"repo": request.repo}, {"issue_type": child_issue_type}, {"parent": parent_key}]},
            include=["metadatas", "documents", "distances"]
        )
        if results and results.get("ids") and results["ids"][0]:
            child_metadata = dict(results["metadatas"][0][0])
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
            matched_metadata = dict(results["metadatas"][0][0])
            matched_metadata["key"] = results["ids"][0][0]
            matched_document = results["documents"][0][0]
            parent_chain.append({k: matched_metadata.get(k, "N/A") for k in ["key", "summary", "issue_type"]})
            parent_key = matched_metadata.get("key")
            for child_issue_type in search_priority[i+1:]:
                child_meta, child_doc = find_best_descendant(parent_key, child_issue_type)
                if child_meta:
                    parent_chain.append({k: child_meta.get(k, "N/A") for k in ["key", "summary", "issue_type"]})
                    matched_metadata, matched_document, parent_key = child_meta, child_doc, child_meta.get("key")
                else: break
            break

    hierarchy_view = "No direct parent hierarchy found."
    if parent_chain:
        seen_keys = set()
        for p in reversed(parent_chain):
            if p['key'] not in seen_keys:
                unique_chain.insert(0, p)
                seen_keys.add(p['key'])
        hierarchy_view = "Hierarchy of Related Ticket:\n" + "\n".join([f"- {p['issue_type']}: {p['key']} - {p['summary']}" for p in unique_chain])

    ticket_generation_system_instruction = """
You are an expert JIRA Ticket Automation Agent. Your primary function is to analyze a developer's commit message, which represents completed technical work, and generate an appropriate JIRA ticket.

**Detailed Jira Issue Type Definitions:**

- **Task:**
  - **Definition:** A Task is a specific, self-contained unit of technical work that must be completed to achieve a larger goal, typically as part of a Story. It is not typically written from an end-user's perspective but rather describes a technical action the development team needs to take.
  - **Purpose:** To track the individual technical steps required to build a feature.
  - **Examples:** "Create the API endpoint for image uploads," "Configure the new database schema," "Set up the CI/CD pipeline for the new service."
  - **Relationship:** Tasks can be children of a Story or can exist independently for general technical work that doesn't map to a specific user-facing feature (e.g., "Upgrade server memory").

- **Bug:**
  - **Definition:** A Bug is a flaw, error, or defect in the software that causes it to produce an incorrect or unexpected result, or to behave in unintended ways. It represents a deviation from the expected or specified behavior.
  - **Purpose:** To track and prioritize the fixing of defects in the application. A bug is reactive; it addresses something that is broken.
  - **Examples:** "User cannot log in with a valid password," "Checkout button is unresponsive on mobile devices," "Incorrect sales tax is calculated for international orders."
  - **Relationship:** Bugs are typically standalone issues but can be linked to the Stories or Epics they affect.

- **Sub-task:**
  - **Definition:** A Sub-task is the most granular level of work tracking, representing a smaller, manageable piece of a parent issue (which can be a Story, Task, or even a Bug).
  - **Purpose:** To break down a larger piece of work into individual steps that can be assigned and tracked separately, providing a clear checklist of what needs to be done to complete the parent issue.
  - **Examples:** For the Task "Create the API endpoint," a Sub-task could be "Write unit tests for the upload endpoint" or "Add validation for file types."
  - **Relationship:** A Sub-task cannot exist on its own; it must always have a parent issue.

**CRITICAL RULE: YOUR ENTIRE RESPONSE MUST BE A SINGLE, VALID JSON OBJECT THAT STRICTLY ADHERES TO THE FOLLOWING SCHEMA. DO NOT INCLUDE ANY EXTRA FIELDS NOT LISTED HERE.**
The JSON MUST include ALL these keys, and ONLY these keys:
{{
    "summary": "Concise and professional Jira Ticket Summary (string)",
    "description": "A validation instruction for a QA tester. (string)",
    "issue_type": "Task" or "Bug" or "Sub-task" (string)",
    "parent": "KEY_OF_PARENT_TICKET" (string) or "N/A"
}}

**Rules for Generating Tickets:**
1.  **Strict Adherence to Schema:** You MUST generate a JSON object that exactly matches the provided schema. This means including ALL specified keys (`summary`, `description`, `issue_type`, `parent`) and **EXCLUDING ANY OTHER KEYS**.
2.  **Summary:** Create a concise, professional summary for the ticket. This field is **MANDATORY**.
    * Example: "Test Google OAuth Login Functionality"
3.  **Description:** The commit message describes work that is **already completed**. Your task is to write a description that instructs a tester on how to **validate** this completed work.
    * For **'Task'** or **'Bug'** issue types, the description **MUST START** with the word "Test" or "Verify". It must not describe how to implement the feature, only what to check.
        * Example (Commit: "feat: Added Google OAuth to login"): "Test the Google OAuth integration on the login page to ensure it functions correctly."
        * Example (Commit: "fix: Login button not clickable"): "Verify that the login button is now clickable and functions as expected on iOS."
    * For **'Sub-task'**, the description should be a direct statement of the technical work that was completed.
        * Example: "Backend API endpoint for password reset was implemented."
4.  **Issue Type:** You MUST select one of the following exact types: **"Task", "Bug", or "Sub-task"**. This field is **MANDATORY**.
5.  **Parent:** Use the key of the most direct parent from the provided 'Hierarchy of Related Ticket'. If no parent is found, set this to "N/A". This field is **MANDATORY**.

**IMPORTANT:** The "Related Ticket" is provided only for *contextual understanding*. It is **NOT** a template for the JSON structure. Your output JSON MUST only contain `summary`, `description`, `issue_type`, and `parent`.
"""
    
    # --- THIS VARIABLE DEFINITION WAS MISSING ---
    ticket_generation_user_prompt = f"""
    New Ticket Context:
    Commit Message: {request.commit_message}
    Repository: {request.repo}
    Related Ticket (for context):
    {matched_document if matched_document else "N/A"}
    Metadata: {json.dumps(matched_metadata, indent=2) if matched_metadata else "N/A"}
    {hierarchy_view}
    Generate the JIRA ticket as a JSON object.
    """
    # --- END OF FIX ---

    try:
        logging.info("Calling Ticket Generation LLM (Gemini).")
        ticket_model = GenerativeModel(
            "gemini-2.5-pro",
            system_instruction=[ticket_generation_system_instruction]
        )
        response = ticket_model.generate_content(
            [ticket_generation_user_prompt],
            generation_config={"temperature": 0.5, "response_mime_type": "application/json"},
            safety_settings={
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
        )
        parsed_json = json.loads(response.text.strip())
        
        required_keys = ["summary", "description", "issue_type", "parent"]
        if not all(key in parsed_json for key in required_keys):
            raise ValueError("Generated ticket is missing one or more required keys.")
        if parsed_json["issue_type"] not in ["Task", "Bug", "Sub-task"]:
            raise ValueError(f"Invalid issue_type generated: '{parsed_json['issue_type']}'.")
            
        return {"generated_ticket": parsed_json}
    except Exception as e:
        logging.error(f"Error generating ticket: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during ticket generation: {e}")

@app.post("/generate_client_ticket")
def generate_client_ticket(request: GenerateClientTicketRequest):
    logging.info("Attempting to generate client ticket for: %s", request.request_text)
    
    gatekeeper_system_instruction = """
    You are a requirements analyst AI. Classify a client's request into one of three categories: "Client Feature Request", "Developer Task", or "Vague Request".
    **CRITICAL RULE: You MUST respond ONLY with a single JSON object.**
    **Classification Rules:**
    **A. Client Feature Request (ACCEPT):** Respond ONLY with: `{"status": "Actionable request received. Proceeding with ticket generation."}`
    **B. Developer Task (REJECT):** Respond ONLY with: `{"status": "Ticket not generated. This appears to be a developer task, not a client feature request."}`
    **C. Vague Request (REJECT):** Respond ONLY with: `{"status": "Ticket not generated. The request is too vague. Please provide more details."}`
    """
    gatekeeper_user_prompt = f"Client Request: {request.request_text}"

    try:
        logging.info("Calling Gatekeeper LLM (Gemini) for client request classification.")
        gatekeeper_model = GenerativeModel(
            "gemini-2.5-pro",
            system_instruction=[gatekeeper_system_instruction]
        )
        gatekeeper_response = gatekeeper_model.generate_content(
            [gatekeeper_user_prompt],
            generation_config={"temperature": 0.1, "response_mime_type": "application/json"},
            safety_settings={
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
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
        include=["metadatas", "documents", "distances"]
    )
    related_epic_view, parent_epic_key = "No related Epic found.", "N/A"
    if epic_search_results and epic_search_results.get("ids") and epic_search_results["ids"][0]:
        epic_meta = epic_search_results["metadatas"][0][0]
        parent_epic_key = epic_search_results["ids"][0][0]
        related_epic_view = f"Found a potentially related Epic:\n- {parent_epic_key}: {epic_meta.get('summary')}"

    # --- MODIFIED SYSTEM INSTRUCTION WITH LONG DEFINITIONS ---
    ticket_generation_system_instruction = """
You are an expert JIRA Product Manager AI. Your primary function is to convert a client's natural language request into a well-formed JIRA Epic or Story.

**Detailed Jira Issue Type Definitions:**

- **Epic:**
  - **Definition:** An Epic represents a large, overarching body of work that can be broken down into a number of smaller, manageable stories. It serves as a high-level container for a significant feature, project, or initiative, providing a strategic theme for the work. An Epic is not a single task but rather a collection of related tasks and user stories that, when completed, achieve a significant business objective.
  - **Purpose:** The primary purpose of an Epic is to organize and manage a large project by grouping related stories. Epics help in creating a product roadmap, communicating high-level plans to stakeholders, and tracking progress on major initiatives over time.
  - **Scope:** An Epic is typically too large and complex to be completed in a single sprint and often spans multiple sprints or even quarters. It describes a broad requirement like "User Profile Management" or "Implement a Shopping Cart".

- **Story (User Story):**
  - **Definition:** A Story, or User Story, is a small, self-contained unit of work that delivers tangible value to an end-user. It is a functional requirement written from the user's perspective, not a technical task. It is designed to be a simple, concise description of a feature that captures what a user wants to do and why.
  - **Purpose:** To capture product functionality from the perspective of the user, ensuring that development work is focused on delivering value. It facilitates communication and a shared understanding between the development team and business stakeholders.
  - **Format:** It famously follows the format: "As a [user type], I want [some goal], so that [some benefit]." This structure ensures that the description includes the user, their goal, and the motivation behind it.
  - **Scope:** A Story should be small enough to be fully completed by the team within a single sprint. For example, "As a user, I want to upload a profile picture, so that I can personalize my account."

**CRITICAL RULE: YOUR ENTIRE RESPONSE MUST BE A SINGLE, VALID JSON OBJECT THAT STRICTLY ADHERES TO THE FOLLOWING SCHEMA.**
{{ "summary": "string", "description": "string", "issue_type": "Epic" or "Story", "parent": "string" or "N/A" }}

**Rules for Generating Tickets:**
1.  **Analyze the Request:** Based on the definitions, determine if the request describes a large Epic or a specific Story.
2.  **Epic Summary Style:** If you create an **Epic**, the `summary` MUST be a short, high-level title, typically 2-4 words. It should name a major feature area.
    - **Good examples:** 'User Authentication', 'Shopping Cart', 'Product Search'.
    - **Bad example:** 'Implement a new system for users to log in and out of their accounts'.
3.  **Story Description Format:** If you choose **Story**, the `description` **MUST** be in the user story format ("As a [user type]...").
4.  **User Type Restriction:** The `[user type]` **MUST be one of**: 'user', 'client', 'mobile user', or 'desktop user'.
5.  **Parent Logic:** An **Epic**'s parent is always "N/A". A **Story**'s parent should be the key of a related Epic from the context.
"""
    ticket_generation_user_prompt = f"Client's Request: \"{request.request_text}\"\n\nContext:\n{related_epic_view}\n\nGenerate the JIRA ticket as a JSON object."
    try:
        logging.info("Calling Ticket Generation LLM (Gemini) for client ticket.")
        ticket_model = GenerativeModel(
            "gemini-2.5-pro",
            system_instruction=[ticket_generation_system_instruction]
        )
        response = ticket_model.generate_content(
            [ticket_generation_user_prompt],
            generation_config={"temperature": 0.6, "response_mime_type": "application/json"},
            safety_settings={
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
        )
        parsed_json = json.loads(response.text.strip())
        
        required_keys = ["summary", "description", "issue_type", "parent"]
        if not all(key in parsed_json and parsed_json[key] is not None for key in required_keys):
            raise ValueError("Generated ticket is missing required keys or they are null.")
        if parsed_json["issue_type"] not in ["Epic", "Story"]:
            raise ValueError(f"Invalid issue_type generated: '{parsed_json['issue_type']}'.")
        if parsed_json["issue_type"] == "Epic":
            parsed_json["parent"] = "N/A"
        elif parsed_json["issue_type"] == "Story" and not parsed_json.get("parent"):
             parsed_json["parent"] = parent_epic_key
             
        return {"generated_ticket": parsed_json}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during ticket generation: {e}")

@app.post("/clear_collection")
def clear_collection():
    """
    Clears all documents from the ChromaDB collection. Use with caution.
    """
    try:
        num_items = collection.count()
        if num_items == 0:
            logging.info("Collection was already empty. No documents to clear.")
            return {"status": "Collection cleared"}

        all_items = collection.get(limit=num_items)
        all_ids = all_items.get("ids")

        if all_ids:
            collection.delete(ids=all_ids)
            logging.info(f"Cleared {len(all_ids)} documents from collection.")
        else:
            logging.info("Collection contained no items to clear despite non-zero count.")
        
        return {"status": "Collection cleared"}
    except Exception as e:
        logging.error(f"An error occurred while clearing the collection: {e}")
        raise HTTPException(status_code=500, detail=str(e))
