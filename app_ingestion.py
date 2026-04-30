import copy
import os
import tempfile
import openpyxl
import streamlit as st
import pandas as pd
import json
from EYQ_chat import generateChatResponse
from AICOE import createPrompt

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
    Upload an Excel file containing test case generation inputs.
    The file should have columns: center, domain_context_query, requirement_context_query, user_story_json, existing_test_cases_json, bugs_json.
    """)

    uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx"])

    if uploaded_file is not None:
        try:
            df = pd.read_excel(uploaded_file)
            st.write("Excel file loaded successfully.")
            st.write(f"Columns found: {df.columns.tolist()}")

            if 'center' not in df.columns:
                st.error("The Excel file must contain a 'center' column.")
                return

            centers = df['center'].dropna().unique().tolist()
            if not centers:
                st.error("No centers found in the 'center' column.")
                return

            selected_center = st.selectbox("Select Center to generate test cases for", options=centers)

            num_test_cases = st.number_input(
                "Number of test cases to generate",
                min_value=1,
                max_value=50,
                value=10,
                step=1
            )

            if st.button("Generate Test Cases"):
                with st.spinner("Generating test cases with LLM... Please wait ⏳"):
                    try:
                        filtered_df = df[df['center'] == selected_center]

                        for idx, row in filtered_df.iterrows():
                            prompt = createPrompt(
                                center=row['center'],
                                domain_context_query=row['domain_context_query'],
                                requirement_context_query=row['requirement_context_query'],
                                user_story_json=row['user_story_json'],
                                existing_test_cases_json=row['existing_test_cases_json'],
                                bugs_json=row['bugs_json'],
                                num_test_cases=num_test_cases
                            )

                            response_raw = generateChatResponse(prompt)

                            response_clean = response_raw.strip()
                            if response_clean.startswith("```"):
                                response_clean = response_clean.strip("```json").strip("```").strip()

                            test_cases = json.loads(response_clean)

                            display_cases = copy.deepcopy(test_cases)
                            for tc in display_cases:
                                tc["test_steps"] = _format_steps_for_display(tc.get("test_steps"))
                                tc["test_steps_summary"] = _format_steps_for_display(tc.get("test_steps_summary"))

                            df_display = pd.DataFrame(display_cases)

                            st.success(f"✅ Test cases generated successfully for center: {selected_center}!")

                            st.dataframe(df_display, use_container_width=True, height=400)

                            col1, col2 = st.columns(2)
                            with col1:
                                st.download_button(
                                    label="⬇️ Download JSON (original)",
                                    data=json.dumps(test_cases, indent=2),
                                    file_name=f"generated_test_cases_{selected_center}.json",
                                    mime="application/json"
                                )
                            with col2:
                                csv_bytes = df_display.to_csv(index=False).encode("utf-8")
                                st.download_button(
                                    label="⬇️ Download CSV (table view)",
                                    data=csv_bytes,
                                    file_name=f"generated_test_cases_{selected_center}.csv",
                                    mime="text/csv"
                                )

                    except Exception as e:
                        st.error(f"🔥 Error during generation: {e}")

        except Exception as e:
            st.error(f"❌ Failed to read Excel file: {e}")
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