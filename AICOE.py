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
    return retriever.invoke(query)


def createPrompt(center, domain_context_query, requirement_context_query, user_story_json,
                 existing_test_cases_json, bugs_json, num_test_cases=10):
    print(f"Fetching {center} domain knowledge...")
    domain_docs = fetch_context(
        collection_name=f"{center.lower()}_domain_contexts",
        subdir=f"{center.lower()}_domain_knowledge",
        top_k=10,
        query=domain_context_query
    )
    domain_context = "\n\n".join(doc.page_content for doc in domain_docs)

    domain_context_summary = generateChatResponse(
        f"""
        You are a {center} SME.
        Summarize all key lifecycle steps and business rules relevant to the domain.
        Output plain text only.

        Data:
        {domain_context}
        """
    )

    print(f"Fetching {center} requirements...")
    req_docs = fetch_context(
        collection_name=f"{center.lower()}_requirement_contexts",
        subdir=f"{center.lower()}_requirements",
        top_k=5,
        query=requirement_context_query
    )
    req_doc_final = "\n\n".join(doc.page_content for doc in req_docs)

    user_story = json.loads(user_story_json)
    existing_test_cases = json.loads(existing_test_cases_json)
    bugs = json.loads(bugs_json)

    prompt = f"""
    You are an expert Domain SME and Test Case Generator specializing in {center}.

    Identify at least {num_test_cases} missing or uncovered test cases related to key workflows.

    Each test case must include:
    - test_case_id
    - requirement_id
    - title (end-to-end workflow oriented)
    - objective
    - preconditions
    - detailed test_steps with clear, step-by-step instructions (avoid generic steps)
    - test_steps_summary
    - expected_results
    - actual_results ("To be tested")
    - test_type

    Rules:
    1. Do NOT repeat existing test cases or known bugs.
    2. Derive logical steps from domain knowledge when needed.
    3. Include negative, boundary, integration, and financial scenarios.
    4. Follow real-world insurance standards.
    5. Output VALID JSON ONLY.

    Input:
    {{
        "domain_knowledge": {json.dumps(domain_context_summary)},
        "user_story": {json.dumps(user_story)},
        "use_case_requirements": {json.dumps(req_doc_final)},
        "existing_test_cases": {json.dumps(existing_test_cases)},
        "bugs": {json.dumps(bugs)}
    }}
    """

    return prompt


def generate_prompts_from_excel(file_path, num_test_cases=10):
    df = pd.read_excel(file_path)
    prompts = {}
    for idx, row in df.iterrows():
        center = row['center']
        prompt = createPrompt(
            center=center,
            domain_context_query=row['domain_context_query'],
            requirement_context_query=row['requirement_context_query'],
            user_story_json=row['user_story_json'],
            existing_test_cases_json=row['existing_test_cases_json'],
            bugs_json=row['bugs_json'],
            num_test_cases = num_test_cases
        )
        prompts[center] = prompt
    return prompts


# Example usage:
# prompts = generate_prompts_from_excel("center_prompts.xlsx")
# for center, prompt in prompts.items():
#     print(f"Prompt for {center}:\n{prompt}\n\n")
