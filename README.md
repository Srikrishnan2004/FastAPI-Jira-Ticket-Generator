# FastAPI-Jira-Ticket-Generator

A FastAPI-based service for generating, classifying, and managing Jira tickets using AI and semantic search. Integrates with Jira, Google Vertex AI (Gemini), and ChromaDB for advanced ticket automation and semantic document storage.

## Features
- Generate Jira tickets from commit messages or client requests
- Classify and log ticket generation and approval decisions
- Store and search documents semantically using a local sentence transformer and ChromaDB
- Integrate with Google Vertex AI for generative tasks
- RESTful API endpoints for ticket and document management

## Requirements
- Python 3.9+
- MySQL database
- Jira account and API token
- Google Cloud account (for Vertex AI)
- [ChromaDB](https://www.trychroma.com/) for vector storage
- [Sentence Transformers](https://www.sbert.net/) model files (provided in `local_sentence_transformer/`)

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd FastAPI-Jira-Ticket-Generator
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare the local sentence transformer model:**
   The directory `local_sentence_transformer/` should contain all necessary model files. If missing, download a compatible model (e.g., `all-MiniLM-L6-v2`) and place its files here.

4. **Set up environment variables:**
   Create a `.env` file in the project root with the following variables:
   ```env
   jira_email=your_jira_email
   jira_api_token=your_jira_api_token
   jira_base_url=https://your-domain.atlassian.net
   project_key=YOUR_PROJECT_KEY
   epic_link_field_id=customfield_XXXXX
   db_user=your_db_user
   db_password=your_db_password
   db_host=your_db_host
   db_name=your_db_name
   GCP_PROJECT_ID=your_gcp_project_id
   GCP_LOCATION=your_gcp_location
   ```

## Running the API

```bash
uvicorn chroma_api_gemini:app --reload
```

The API will be available at `http://localhost:8000` by default.

## Docker

A `Dockerfile` is provided for containerized deployment:

```bash
docker build -t fastapi-jira-ticket-generator .
docker run --env-file .env -p 8000:8000 fastapi-jira-ticket-generator
```

## API Endpoints

- `POST /add` — Add a document to ChromaDB
- `GET /getall` — Retrieve all documents
- `POST /generate_ticket` — Generate a Jira ticket from a commit message
- `POST /generate_client_ticket` — Generate a Jira ticket from a client request
- `POST /clear_collection` — Clear the ChromaDB collection

See the code in `chroma_api_gemini.py` for request/response schemas and additional endpoints.

## Model Details

The local sentence transformer model is based on `all-MiniLM-L6-v2` and is used for semantic search and embedding generation. See `local_sentence_transformer/README.md` for more details and usage examples.

## License

This project is licensed under the Apache 2.0 License. 