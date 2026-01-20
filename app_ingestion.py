import copy
import os
import tempfile

import streamlit as st
import pandas as pd
import json
from EYQ_chat import generateChatResponse
from AICOE import createPrompt
from ingestion_script_EYQ import ingest_domain_knowledge_to_chroma, ingest_requirement_docs_to_chroma


# ============================================================
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
                st.error(f"❌ Ingestion failed: {e}")
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
    Click below to generate new test cases using the domain and requirement knowledge base.
    """)

    if st.button("Generate Test Cases"):
        with st.spinner("Generating test cases with LLM... Please wait ⏳"):
            try:
                # 1️⃣ Call your existing logic
                response_raw = generateChatResponse(createPrompt())

                # 2️⃣ Clean up any markdown fences (```json)
                response_clean = response_raw.strip()
                if response_clean.startswith("```"):
                    response_clean = response_clean.strip("```json").strip("```").strip()

                # 3️⃣ Try parsing JSON
                try:
                    test_cases = json.loads(response_clean)
                except Exception as e:
                    st.error(f"❌ JSON parsing failed: {e}")
                    st.text_area("Raw LLM Output", response_raw, height=400)
                    return

                # 4️⃣ Validate structure
                if not isinstance(test_cases, list):
                    st.warning("⚠️ Output is not a list of test cases. Showing raw content:")
                    st.json(test_cases)
                    return

                # Keep original JSON for download (unaltered)
                original_json = copy.deepcopy(test_cases)

                # Create a copy for display/CSV where only test_steps are formatted
                display_cases = copy.deepcopy(test_cases)
                for tc in display_cases:
                    # Normalize and format test_steps into numbered multiline string
                    tc_steps = tc.get("test_steps", None)
                    tc["test_steps"] = _format_steps_for_display(tc_steps)

                for tc in display_cases:
                    # Normalize and format test_steps into numbered multiline string
                    tc_steps_summary = tc.get("test_steps_summary", None)
                    tc["test_steps_summary"] = _format_steps_for_display(tc_steps_summary)

                # Create DataFrame for tabular display
                df = pd.DataFrame(display_cases)

                st.success("✅ Test cases generated successfully!")

                # Display the DataFrame (tabular). test_steps will show numbered multiline text in each cell.
                st.dataframe(df, use_container_width=True, height=400)

                # Download buttons
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        label="⬇️ Download JSON (original)",
                        data=json.dumps(original_json, indent=2),
                        file_name="generated_test_cases.json",
                        mime="application/json"
                    )
                with col2:
                    # CSV from the display dataframe will contain the numbered multiline steps
                    csv_bytes = df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="⬇️ Download CSV (table view)",
                        data=csv_bytes,
                        file_name="generated_test_cases.csv",
                        mime="text/csv"
                    )

            except Exception as e:
                st.error(f"🔥 Error during generation: {e}")


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