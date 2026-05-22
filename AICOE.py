import json
import os
import pandas as pd

import chromadb
from chromadb.config import Settings

from langchain_chroma import Chroma
from langchain_openai import AzureOpenAIEmbeddings

from EYQ_chat import generateChatResponse


# Vector DB Configuration
PERSIST_DIR = "./vectordb"
CHROMA_TENANT = os.getenv("CHROMA_TENANT") or "default_tenant"
CHROMA_DATABASE = os.getenv("CHROMA_DATABASE") or "default_database"


def load_retriever(collection_name, subdir, top_k):
    persist_path = os.path.join(PERSIST_DIR, subdir)

    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=os.getenv("AZURE_EMBEDDING_DEPLOYMENT") or "text-embedding-3-large",
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        or "https://eyq-incubator.america.fabric.ey.com/eyq/us/api",
        api_key=os.getenv("OPENAI_KEY") or os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION") or "2025-04-01-preview",
    )

    try:
        client = chromadb.PersistentClient(
            path=persist_path,
            settings=Settings(anonymized_telemetry=False),
            tenant=CHROMA_TENANT,
            database=CHROMA_DATABASE,
        )
    except Exception:
        client = chromadb.PersistentClient(
            path=persist_path,
            settings=Settings(anonymized_telemetry=False),
        )

    if hasattr(client, "create_tenant"):
        try:
            client.create_tenant(CHROMA_TENANT)
        except Exception:
            pass

    if hasattr(client, "create_database"):
        try:
            client.create_database(CHROMA_DATABASE, tenant=CHROMA_TENANT)
        except Exception:
            pass

    try:
        client.get_or_create_collection(name=collection_name)
    except Exception:
        pass

    vectordb = Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=embeddings,
    )

    return vectordb.as_retriever(search_kwargs={"k": top_k})


def fetch_context(collection_name, subdir, top_k, query):
    retriever = load_retriever(
        collection_name=collection_name,
        subdir=subdir,
        top_k=top_k
    )
    # Use invoke if available, else fallback to get_relevant_documents
    if hasattr(retriever, "invoke"):
        return retriever.invoke(query)
    elif hasattr(retriever, "get_relevant_documents"):
        return retriever.get_relevant_documents(query)
    else:
        raise AttributeError("Retriever has no method 'invoke' or 'get_relevant_documents'")

    def fetch_context(collection_name, subdir, top_k, query):
        retriever = load_retriever(
            collection_name=collection_name,
            subdir=subdir,
            top_k=top_k
        )
        # Use invoke if available, else fallback to get_relevant_documents
        if hasattr(retriever, "invoke"):
            return retriever.invoke(query)
        elif hasattr(retriever, "get_relevant_documents"):
            return retriever.get_relevant_documents(query)
        else:
            raise AttributeError("Retriever has no method 'invoke' or 'get_relevant_documents'")


def  createPrompt(center):
    center_key = center.lower().replace(" ", "_")  # e.g. claim_center

    # Fetch domain knowledge docs from vector DB
    domain_docs = fetch_context(
        collection_name=f"{center_key}_domain_contexts",
        subdir=f"{center_key}_domain_knowledge",
        top_k=10,
        query=""  # empty or a general query to get top domain docs
    )

    domain_context = "\n\n".join(doc.page_content for doc in domain_docs)

    # Fetch requirement docs from vector DB
    req_docs = fetch_context(
        collection_name=f"{center_key}_requirement_contexts",
        subdir=f"{center_key}_requirements",
        top_k=10,
        query=""  # empty or a general query to get top requirement docs
    )

    requirement_context = "\n\n".join(doc.page_content for doc in req_docs)

    # You can load existing test cases and bugs JSON if available, else empty lists
    existing_test_cases = []
    bugs = []

    # Compose prompt
    prompt = f"""
    You are an expert Domain SME and Test Case Generator specializing in {center}.

    Use the following domain knowledge and requirements to identify missing or uncovered test cases related to key workflows, ensuring maximum coverage.

    Domain Knowledge:
    {domain_context}

    Requirements:
    {requirement_context}

    Existing Test Cases:
    {json.dumps(existing_test_cases)}

    Known Bugs:
    {json.dumps(bugs)}

    Rules:
    1. Do NOT repeat existing test cases or known bugs.
    2. Derive logical steps from domain knowledge when needed.
    3. Include positive, negative, boundary, integration, edge, and financial scenarios.
    4. Follow real-world insurance standards.
    5. Output VALID JSON ONLY with fields: test_case_id, requirement_id, title, objective, preconditions, detailed test_steps, test_steps_summary, expected_results, actual_results, test_type.
    """

    # Call LLM
    response_raw = generateChatResponse(prompt)

    # Process response as before...
    # Return or display generated test cases


def generate_prompts_from_excel(file_path):
    """
    Generate prompts from the original Excel format with columns:
    center, domain_context_query, requirement_context_query, user_story_json, existing_test_cases_json, bugs_json
    """
    df = pd.read_excel(file_path)
    prompts = {}
    for idx, row in df.iterrows():
        center = row.get('center', 'default').strip()
        prompt = createPrompt(
            center=center,
            domain_context_query=row.get('domain_context_query', ''),
            requirement_context_query=row.get('requirement_context_query', ''),
            user_story_json=row.get('user_story_json', '{}'),
            existing_test_cases_json=row.get('existing_test_cases_json', '[]'),
            bugs_json=row.get('bugs_json', '[]')
        )
        if center not in prompts:
            prompts[center] = []
        prompts[center].append(prompt)
    return prompts


def create_prompt_from_guidewire_bdd_row(row):
    """
    Create a prompt for test case generation based on a single Guidewire BDD test case row.
    """
    prompt = f"""
You are an expert Guidewire SME and Test Case Generator specializing in {row.get('Feature', 'General')}.

Given the following test case details, identify missing or uncovered test cases related to key workflows, ensuring maximum coverage.

Test Case ID: {row.get('Test Case ID', '')}
Test Case Name: {row.get('Test Case Name', '')}
Scenario: {row.get('Scenario', '')}
Type: {row.get('Type', '')}
Priority: {row.get('Priority', '')}
Preconditions: {row.get('Preconditions', '')}
Test Data: {row.get('Test Data', '')}
Expected Result: {row.get('Expected Result', '')}

Rules:
1. Do NOT repeat existing test cases.
2. Derive logical steps from domain knowledge when needed.
3. Include positive, negative, boundary, integration, edge, and financial scenarios.
4. Follow real-world insurance standards.
5. Output VALID JSON ONLY with fields: test_case_id, title, objective, preconditions, detailed test_steps, test_steps_summary, expected_results, actual_results, test_type.
"""
    return prompt
