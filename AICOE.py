import json
import os

import chromadb
from chromadb.config import Settings


from langchain_chroma import Chroma
from langchain_openai import AzureOpenAIEmbeddings

from EYQ_chat import generateChatResponse


# ===========================================================
# =============== Load Vector DB =============================
# ===========================================================
PERSIST_DIR = r"./vectordb"

CHROMA_TENANT = os.getenv("CHROMA_TENANT") or "default_tenant"
CHROMA_DATABASE = os.getenv("CHROMA_DATABASE") or "default_database"



# TOP_K = 10

def load_retriever(collection_name, subdir, TOP_K):
    """Loads a retriever for a given Chroma collection (domain or requirements), safely."""
    persist_path = os.path.join(PERSIST_DIR, subdir)

    # --- Embeddings (keep yours; this preserves your current behavior) ---
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=os.getenv("AZURE_EMBEDDING_DEPLOYMENT") or "text-embedding-3-large",
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT") or "https://eyq-incubator.america.fabric.ey.com/eyq/us/api",
        api_key=os.getenv("OPENAI_KEY") or os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION") or "2025-04-01-preview",
        model=os.getenv("AZURE_EMBEDDING_MODEL") or "text-embedding-3-large",
    )

    # --- Create a persistent Chroma client explicitly (prevents tenant errors) ---
    client = chromadb.PersistentClient(
        path=persist_path,
        settings=Settings(anonymized_telemetry=False),
        tenant=CHROMA_TENANT,
        database=CHROMA_DATABASE,
    )

    # --- Auto-create tenant/database if your Chroma version supports it ---
    # (Older versions won't have these methods; this remains safe)
    if hasattr(client, "create_tenant"):
        try:
            client.create_tenant(CHROMA_TENANT)
        except Exception:
            pass  # tenant already exists or not supported by backend

    if hasattr(client, "create_database"):
        try:
            client.create_database(CHROMA_DATABASE, tenant=CHROMA_TENANT)
        except Exception:
            pass  # database already exists or not supported

    # --- Ensure collection exists (important after DB resets) ---
    # This call creates the collection if missing.
    try:
        client.get_or_create_collection(name=collection_name)
    except Exception:
        # If the collection create call fails for any reason,
        # LangChain will attempt to access it; but this reduces failures significantly.
        pass

    # --- Use LangChain Chroma wrapper with the explicit client ---
    vectordb = Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=embeddings,
    )

    return vectordb.as_retriever(search_kwargs={"k": TOP_K})


def fetch_domain_context(question):
    # Load retrievers
    domain_retriever = load_retriever("domain_contexts", "domain_knowledge", 10)
    # Retrieve from both sources
    domain_docs = domain_retriever.invoke(question)
    return domain_docs


def fetch_requirement_context(question):
    """Fetch top K results from both domain knowledge and requirement knowledge bases."""
    # Load retrievers
    req_retriever = load_retriever("requirement_contexts", "requirement_docs", 5)

    # Retrieve from both sources
    req_docs = req_retriever.invoke(question)
    return req_docs


def createPrompt():
    # --- Step 1: Fetch and summarize domain knowledge ---
    print("Fetching domain knowledge...")
    domain_knowledge_base = fetch_domain_context(
        "Create and Bind a New Personal Auto Policy"
    )
    domain_context = "\n\n".join([doc.page_content for doc in domain_knowledge_base])

    # print(domain_context)

    domain_context_summary = generateChatResponse(
        f"You are a GUIDEWIRE POLICY CENTER EXPERT. Please summarize this data but KEEP ALL STEPS such that it can be used as a guide for software testers with important things and validations to keep in mind before testing. Data: {domain_context}. Output should be string text without any special characters so that I can use it for further prompting."
    )

    # --- Step 2: Fetch requirements ---
    print("Fetching requirements...")
    req_doc = fetch_requirement_context("Create and Bind a New Personal Auto Policy")
    req_doc_final = "\n\n".join([doc.page_content for doc in req_doc])

    # --- Step 3: Define base input test data ---
    user_story = {
        "id": "US-201",
        "title": "Home quotation for new business SOR journey should work the same as V8 post the version 10 upgrade",
        "description": (
            "Given PC is upgraded to v10"
"and user is completing a quotation journey for Home Policy"
"When user navigates to Ancillary Coverages screen (this comes as next step after Buildings and Contents)"
"Then the screen should have the same content and behaviour as per v8"
        )
    }

    existing_test_cases = [
        {
            "id": "TC-201",
            "requirement_id": "REQ-201",
            "description": (
                "Verify that a new Personal Auto policy can be created with valid applicant and vehicle details, "
                "and bound successfully."
            )
        },
        {
            "id": "TC-202",
            "requirement_id": "REQ-202",
            "description": (
                "Verify that mandatory fields such as Applicant Name, Address, and Effective Date are validated "
                "during policy creation."
            )
        }
    ]

    bugs = [
        {
            "id": "BUG-201",
            "test_case_id": "TC-201",
            "title": "Policy Binding Error",
            "description": (
                "Policy binding fails with 'Null pointer exception' when applicant address is missing "
                "even though the field is marked optional."
            ),
            "severity": "High",
            "status": "Open"
        },
        {
            "id": "BUG-202",
            "test_case_id": "TC-202",
            "title": "Coverage Premium Calculation Mismatch",
            "description": (
                "Premium is not recalculated after changing coverage limit values during quote revision."
            ),
            "severity": "Medium",
            "status": "In Progress"
        }
    ]

    # --- Step 4: Serialize to JSON strings ---
    domain_context_summary_json = json.dumps(domain_context_summary)
    req_doc_final_json = json.dumps(req_doc_final)
    user_story_json = json.dumps(user_story)
    existing_test_cases_json = json.dumps(existing_test_cases)
    bugs_json = json.dumps(bugs)

    # --- Step 5: Construct prompt --
    prompt = f"""
           You are an expert Domain SME and Test Case Generator specializing in software quality assurance for BANKING and INSURANCE systems.

           Use the provided *domain_knowledge* and *requirements* to deeply understand the business logic, workflows, data rules, validations, edge conditions, and regulatory or compliance needs that are typical for this domain.

           Your task is to identify and generate missing, uncovered, or domain-specific test cases that are NOT already covered by the provided test cases or bugs.

           Each generated test case must include:
           - test_case_id
           - requirement_id
           - title (do not include field by field test case validation, important - generate bind and quote related scenarios which covers end to end workflows)
           - objective
           - preconditions
           - test_steps (detailed, sequential, and realistic — derived from user_story if available, else from domain_context_summary)
           - test_steps_summary (sequential - summarized form of above test_steps omitting details on individual fields) 
           - expected_results (specific measurable outcomes)
           - actual_results (to be filled during execution — leave as "To be tested")
           - test_type (Functional, Integration, Negative, Boundary, Security, etc.)

           Follow these rules carefully:
           1. Analyze domain_knowledge, user_story, use_case_requirements, existing_test_cases, and bugs.
           2. DO NOT repeat any test case, requirement, or bug already listed in the input.
           3. Generate only new, missing, or domain-specific test cases — including edge and negative cases.
           4. If the user_story lacks process details, derive logical steps from domain_context_summary.
           5. Ensure all test cases follow real-world banking and insurance standards (e.g., authentication, policy lifecycle, claim rules, data validation, regulatory compliance).
           6. Include valid expected results and leave "actual_results": "To be tested".
           7. The output must be valid JSON only. No extra commentary, no markdown, no plain text.

           Input:
           {{
             "domain_knowledge": {domain_context_summary_json},
             "user_story": {user_story_json},
             "use_case_requirements": {req_doc_final_json},
             "existing_test_cases": {existing_test_cases_json},
              "bugs": {bugs_json}
           }}
           """
    return prompt

# if __name__ == "__main__":
#
#     # promptGemini(prompt)
#     response_raw = generateChatResponse(createPrompt())
#     print(response_raw)
