import streamlit as st
from phi.agent import Agent
# from phi.model.openai import OpenAIChat
# from phi.model.ollama import Ollama
from phi.model.google import Gemini
from phi.tools.duckduckgo import DuckDuckGo
from google.generativeai import upload_file, get_file
import time
from pathlib import Path
import tempfile

from dotenv import load_dotenv
load_dotenv()

# pg_config
st.set_page_config(
    page_title="Multimodal AI Agent - Video Summarizer",
    page_icon="🎦",
    layout="wide"
)

st.title("Phidata Multmodal AI Agent")
st.header("Powered by Gemini")

@st._cache_resource
def initialize_agent():
    return Agent(
        name="Video AI Summarizer",
        # model=OpenAIChat(id="gpt-4"),
        # model=Ollama(id="michaelneale/deepseek-r1-goose:latest"),
        model=Gemini(id="gemini-2.0-flash-exp"),
        tools=[DuckDuckGo()],
        markdown=True,
        verbose=True,
        show_tool_calls=True
    )


# Initialise the agent
multi_modal_agent = initialize_agent()

# File uploader
video_file = st.file_uploader(
    "Upload a video file", type=["mp4", "mov"], help="Upload a video for AI analysis"
)

if video_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
        temp_video.write(video_file.read())
        video_path = temp_video.name

    st.video(video_path, format="video/mp4", start_time=0)

    user_query = st.text_area(
        "What insights are you seeking from the video?",
        placeholder="Ask anything about the video content. The AI agent will analyze and gather additional information from the internet",
        help="Provide specific questions or insights you want from the video."
    )

    if st.button("Analyse Video", key="analyze_video_button"):
        if not user_query:
            st.warning("Please enter a question or insights from the video.")
        else:
            try:
                with st.spinner("Processing the video and gathering data ..."):
                    # upload and process the file
                    processed_video = upload_file(video_path)
                    while processed_video.state.name == "PROCESSING":
                        time.sleep(1)
                        processed_video = get_file(processed_video.name)

                    # Prompt generation for the analysis
                    analysis_prompt = (
                        f"""
                        Analyse the uploaded video and determine whether its a cooking video or not.
                        If it is a cooking video, then extract all the ingredients and steps to make the particular recipie.
                        Provide the response containing ingredients and steps to cook.
                        
                        Provide a detailed, user-friendly, and actionable response.
                        """
                    )

                    # AI Agent processing
                    response = multi_modal_agent.run(
                        analysis_prompt, videos=[processed_video])

                # Display the result
                st.subheader("Analysis result")
                st.markdown(response.content)

            except Exception as error:
                st.error(f"An error occurred during analysis : {error}")
    else:
        st.info("Upload a video file to begin analysis.")

    # Customize text area height
    st.markdown(
        """
        <style>
        .stTextArea textArea {
            height: 100px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
