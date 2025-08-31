import streamlit as st
import tempfile
import os
import uuid
from RAG_Pipeline.RAG_Pipeline import RAGPipeline

class RAGApp:
    def __init__(self):
        if "uploaded_file_names" not in st.session_state:
            st.session_state.uploaded_file_names = set()
        if "busy" not in st.session_state:
            st.session_state.busy = False
        if "last_answer" not in st.session_state:
            st.session_state.last_answer = ""
        if "collection_name" not in st.session_state:
            st.session_state.collection_name = str(uuid.uuid4())

        self.pipeline = RAGPipeline(embedder_device=-1, collection_name=st.session_state.collection_name)

    # -------------------------------
    # Overlay lock (blocks UI when busy)
    # -------------------------------
    def lock_ui(self, message="⏳ Processing... Please wait"):
        st.markdown(
            f"""
            <div style="
                position: fixed; top: 0; left: 0; width: 100%; height: 100%;
                background-color: rgba(20,20,20,0.75); z-index: 9999;
                display: flex; align-items: center; justify-content: center;
                flex-direction: column;
                color: white; font-size: 26px; font-weight: 600;">
                <div class="loader"></div>
                {message}
            </div>
            <style>
            .loader {{
                border: 8px solid #555;
                border-top: 8px solid #03dac6;
                border-radius: 50%;
                width: 60px;
                height: 60px;
                animation: spin 1s linear infinite;
                margin-bottom: 20px;
            }}
            @keyframes spin {{
                0% {{ transform: rotate(0deg); }}
                100% {{ transform: rotate(360deg); }}
            }}
            </style>
            """,
            unsafe_allow_html=True
        )

    # -------------------------------
    # Main Layout
    # -------------------------------
    def run(self):
        st.set_page_config(page_title="Advanced RAG")  # narrow layout
        st.markdown("<h1 style='text-align: center; color: white;'>📚 Advanced RAG Web Service</h1>", unsafe_allow_html=True)

        st.markdown("""
        <style>
        body {
            background-color: #121212;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            color: white;
        }
        .stApp {
            background-color: #1e1e1e;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.3);
        }
        .stButton>button {
            background-color: #6200ea;
            color: white;
            border: none;
            border-radius: 5px;
            padding: 10px 20px;
            font-size: 16px;
        }
        .stButton>button:hover {
            background-color: #7c4dff;
        }
        .stTextInput>div>div>input {
            background-color: #2e2e2e;
            color: white;
            border-radius: 5px;
            border: 1px solid #555;
            padding: 10px;
        }
        .stFileUploader>div>div>div>button {
            background-color: #03dac6;
            color: #121212;
            border: none;
            border-radius: 5px;
            padding: 10px 20px;
        }
        .stFileUploader>div>div>div>button:hover {
            background-color: #00bfa5;
        }
        .stSubheader {
            color: white;
        }
        .stInfo {
            background-color: #333;
            color: white;
        }
        .stSuccess {
            background-color: #4caf50;
            color: white;
        }
        .stError {
            background-color: #f44336;
            color: white;
        }
        .stWarning {
            background-color: #ff9800;
            color: white;
        }
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown p, .stMarkdown li {
            color: white;
        }
        </style>
        """, unsafe_allow_html=True)

        # Overlay lock
        if st.session_state.busy:
            self.lock_ui("🔄 Processing your request...")
            self.render_upload_section()
        else:
            # Two-column layout
            col1, col2 = st.columns([1, 2])

            with col1:
                self.render_upload_section()
                self.render_clear_button()

            with col2:
                self.render_qa_section()

    # -------------------------------
    # Upload Section
    # -------------------------------
    def render_upload_section(self):
        if st.session_state.busy:
            st.info("🔄 Processing file... Please wait.")
        else:
            st.subheader("📂 Upload Document")

            uploaded_file = st.file_uploader(
                "Select a document",
                type=["pdf", "docx", "txt"],
                accept_multiple_files=False,
                label_visibility="collapsed"
            )

            if uploaded_file and uploaded_file.name not in st.session_state.uploaded_file_names:
                st.session_state.busy = True
                try:
                    with st.spinner(f"Processing {uploaded_file.name}..."):
                        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                            tmp_file.write(uploaded_file.read())
                            tmp_path = tmp_file.name

                        self.pipeline.ingest_pdf(tmp_path)

                    st.session_state.uploaded_file_names.add(uploaded_file.name)
                    st.success(f"✅ {uploaded_file.name} added to database")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                finally:
                    st.session_state.busy = False
                    # NO st.rerun() here — keep overlay until next user interaction

    # -------------------------------
    # Clear DB
    # -------------------------------
    def render_clear_button(self):
        st.markdown("---")
        if st.button("🗑️ Clear Database", use_container_width=True, disabled=st.session_state.busy):
            st.session_state.busy = True
            try:
                with st.spinner("Clearing Qdrant database..."):
                    self.pipeline.qdrant_handler.delete_collection()
                st.session_state.uploaded_file_names.clear()
                st.session_state.last_answer = ""
                st.success("✅ Database cleared")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
            finally:
                st.session_state.busy = False
                # NO st.rerun() here — keep overlay until next user interaction

    # -------------------------------
    # QA Section
    # -------------------------------
    def render_qa_section(self):
        st.subheader("💡 Ask Questions")

        # Permanent answer box
        if st.session_state.last_answer:
            st.markdown("### 📝 Answer:")
            st.markdown(st.session_state.last_answer)
            st.markdown("---")

        question = st.text_input(
            "Type your question here...",
            disabled=st.session_state.busy,
            placeholder="e.g. What is mentioned in the uploaded report?"
        )

        if st.button("🤖 Get Answer", use_container_width=True, disabled=st.session_state.busy):
            if question.strip():
                st.session_state.busy = True
                try:
                    with st.spinner("Generating answer..."):
                        answer = self.pipeline.ask(question)
                    st.session_state.last_answer = answer
                    st.success("✅ Answer ready")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                finally:
                    st.session_state.busy = False
                    st.rerun()
            else:
                st.warning("⚠️ Please enter a question first.")


# -------------------------------
# Run App
# -------------------------------
if __name__ == "__main__":
    app = RAGApp()
    app.run()
