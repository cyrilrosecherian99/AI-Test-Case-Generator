import copy
import os
import tempfile

import streamlit as st
import pandas as pd
import json
from EYQ_chat import generateChatResponse
from AICOE import createPrompt, create_prompt_from_guidewire_bdd_row
from AICOE import fetch_context


from ingestion_script_EYQ import ingest_domain_knowledge_to_chroma, ingest_requirement_docs_to_chroma



# ===========================================================
# INGESTION TAB
# ============================================================

def ingestion_tab():
    st.header("📂 Data Ingestion")

    st.write("""
    Upload your **Domain Knowledge** and/or **Requirement Documents** below (optional).  
    If you don't have any files, you can still proceed without uploading.
    These will be processed, chunked, and stored in the Chroma vector database if provided.
    """)

    # ---- Domain Documents Upload ----
    st.subheader("📘 Domain Knowledge Files (Optional)")
    domain_files = st.file_uploader(
        "Upload Domain Documents (PDF, DOC, DOCX) – Optional",
        type=["pdf", "doc", "docx"],
        accept_multiple_files=True
    )

    # ---- Requirement Documents Upload ----
    st.subheader("📄 Requirement Documents (Optional)")
    requirement_files = st.file_uploader(
        "Upload Requirement Documents (PDF, DOC, DOCX) – Optional",
        type=["pdf", "doc", "docx"],
        accept_multiple_files=True
    )

    if st.button("🚀 Run Ingestion"):
        with st.spinner("Running ingestion... Please wait ⏳"):
            try:
                with tempfile.TemporaryDirectory() as temp_dir:
                    domain_folder = os.path.join(temp_dir, "domain_docs")
                    req_folder = os.path.join(temp_dir, "requirement_docs")
                    os.makedirs(domain_folder, exist_ok=True)
                    os.makedirs(req_folder, exist_ok=True)

                    # Save uploaded domain files
                    domain_paths = []
                    if domain_files:
                        for f in domain_files:
                            file_path = os.path.join(domain_folder, f.name)
                            with open(file_path, "wb") as out_file:
                                out_file.write(f.read())
                            domain_paths.append(file_path)

                    # Save uploaded requirement files
                    req_paths = []
                    if requirement_files:
                        for f in requirement_files:
                            file_path = os.path.join(req_folder, f.name)
                            with open(file_path, "wb") as out_file:
                                out_file.write(f.read())
                            req_paths.append(file_path)

                    # ---- Call ingestion functions only if files exist ----
                    if domain_paths:
                        st.info(f"Ingesting {len(domain_paths)} domain documents...")
                        ingest_domain_knowledge_to_chroma(domain_folder)

                    if req_paths:
                        st.info(f"Ingesting {len(req_paths)} requirement documents...")
                        ingest_requirement_docs_to_chroma(req_paths)

                if not domain_paths and not req_paths:
                    st.warning("⚠️ No files were uploaded. Nothing to ingest.")
                else:
                    st.success("✅ Ingestion completed successfully!")

            except Exception as e:
                import traceback
                st.error(f"❌ Ingestion failed: {e}")
                st.code(traceback.format_exc())


# ============================================================
# TEST CASE GENERATION TAB
# ============================================================

def _format_steps_for_display(steps):
    """
    Convert steps (list or string) into a numbered multiline string.
    """
    if steps is None:
        return ""

    # If it's already a list of steps
    if isinstance(steps, list):
        cleaned = [s.strip() for s in steps if s is not None]
    else:
        # If it's a single string, split into lines and clean
        cleaned = [line.strip() for line in str(steps).splitlines() if line.strip()]

    # Number them
    numbered = [f"{i+1}. {line}" for i, line in enumerate(cleaned)]
    return "\n".join(numbered)


def test_case_generation_tab():
    st.header("🧪 AI Test Case Generation")

    st.write("""
    Upload an Excel file containing prompt inputs per Guidewire center.
    Columns expected: center, user_story_title, user_story_description, existing_test_case_id, existing_test_case_title, bug_id, bug_description.
    Select a Guidewire Center to generate test cases based on ingested domain and requirement documents.
    """)

    uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx"])

    centers = ["Policy Center", "Billing Center", "Claim Center"]
    selected_center = st.selectbox("Select Guidewire Center", options=centers)

    if uploaded_file is not None:
        try:
            df = pd.read_excel(uploaded_file)

            # Required columns for new format
            required_cols = [
                'center',
                'user_story_title',
                'user_story_description',
                'existing_test_case_id',
                'existing_test_case_title',
                'bug_id',
                'bug_description'
            ]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                st.error(f"Uploaded Excel is missing required columns: {missing_cols}")
                return

            # Filter rows for selected center
            filtered_rows = df[df['center'].str.lower() == selected_center.lower()]
            if filtered_rows.empty:
                st.warning(f"No prompt inputs found for center '{selected_center}'.")
                return

            all_generated_test_cases = []

            with st.spinner(f"Generating test cases for center '{selected_center}'... Please wait ⏳"):
                for idx, row in filtered_rows.iterrows():
                    # Fetch domain knowledge from vector DB
                    domain_docs = fetch_context(
                        collection_name=f"{selected_center.lower().replace(' ', '_')}_domain_contexts",
                        subdir=f"{selected_center.lower().replace(' ', '_')}_domain_knowledge",
                        top_k=10,
                        query=""  # no query needed, fetch top docs
                    )
                    domain_context = "\n\n".join(doc.page_content for doc in domain_docs)

                    # Fetch requirement docs from vector DB
                    req_docs = fetch_context(
                        collection_name=f"{selected_center.lower().replace(' ', '_')}_requirement_contexts",
                        subdir=f"{selected_center.lower().replace(' ', '_')}_requirements",
                        top_k=10,
                        query=""
                    )
                    requirement_context = "\n\n".join(doc.page_content for doc in req_docs)

                    # Build JSON objects from plain text columns
                    user_story = {
                        "title": row['user_story_title'],
                        "description": row['user_story_description']
                    }

                    existing_test_cases = []
                    if pd.notna(row['existing_test_case_id']) and row['existing_test_case_id'].strip():
                        existing_test_cases.append({
                            "test_case_id": row['existing_test_case_id'],
                            "title": row['existing_test_case_title']
                        })

                    bugs = []
                    if pd.notna(row['bug_id']) and row['bug_id'].strip():
                        bugs.append({
                            "bug_id": row['bug_id'],
                            "description": row['bug_description']
                        })

                    # Build prompt
                    prompt = f"""
You are an expert Domain SME and Test Case Generator specializing in {selected_center}.

Identify missing or uncovered test cases related to key workflows, ensuring maximum coverage.

Domain Knowledge:
{domain_context}

Requirements:
{requirement_context}

User Story:
{json.dumps(user_story)}

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

                    response_clean = response_raw.strip()
                    if response_clean.startswith("```"):
                        response_clean = response_clean.strip("```json").strip("```").strip()

                    test_cases = json.loads(response_clean)

                    # Format steps for display
                    for tc in test_cases:
                        tc["test_steps"] = _format_steps_for_display(tc.get("test_steps"))
                        tc["test_steps_summary"] = _format_steps_for_display(tc.get("test_steps_summary"))

                    all_generated_test_cases.extend(test_cases)

            if not all_generated_test_cases:
                st.warning("No test cases were generated.")
                return

            df_display = pd.DataFrame(all_generated_test_cases)

            st.success(f"✅ Generated {len(all_generated_test_cases)} test cases for center '{selected_center}'!")

            st.dataframe(df_display, use_container_width=True, height=400)

            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    label="⬇️ Download JSON (original)",
                    data=json.dumps(all_generated_test_cases, indent=2),
                    file_name=f"generated_test_cases_{selected_center.replace(' ', '_').lower()}.json",
                    mime="application/json"
                )
            with col2:
                csv_bytes = df_display.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇️ Download CSV (table view)",
                    data=csv_bytes,
                    file_name=f"generated_test_cases_{selected_center.replace(' ', '_').lower()}.csv",
                    mime="text/csv"
                )

        except Exception as e:
            st.error(f"❌ Failed to process Excel file or generate test cases: {e}")

    else:
        st.info("Please upload an Excel file to proceed.")



# ============================================================
# MAIN APP
# ============================================================

def main():
    st.set_page_config(page_title="EYQ AI Test Case Generator", layout="wide")
    st.title("EYQ AI – End-to-End Test Case Automation")
    tab1, tab2 = st.tabs(["📂 Ingestion", "🧪 Test Case Generation"])

    with tab1:
        ingestion_tab()

    with tab2:
        test_case_generation_tab()


if __name__ == "__main__":
    main()