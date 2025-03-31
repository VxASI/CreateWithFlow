import streamlit as st
import os
import json
import tempfile
import time
import logging
import sys
import traceback
from dotenv import load_dotenv
from streamlit.runtime.scriptrunner import get_script_run_ctx, add_script_run_ctx
from streamlit.runtime.scriptrunner import ScriptRunContext

# Set up script run context for background threads
def get_or_create_script_run_ctx():
    ctx = get_script_run_ctx()
    if ctx is None:
        # Create and set a new context if none exists
        ctx = ScriptRunContext(
            session_id="_background_",
            enqueue=lambda fn: fn(),
            query_string="",
            session_state={},
            initial_page_script_hash="",
            page_script_hash="",
        )
        add_script_run_ctx(ctx)
    return ctx

# Configure logging more verbosely
logging.basicConfig(level=logging.DEBUG, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                   handlers=[logging.StreamHandler(), logging.FileHandler("app_debug.log")])
logger = logging.getLogger(__name__)

# Add exception hook to catch unhandled exceptions
def handle_exception(exc_type, exc_value, exc_traceback):
    logger.error("Unhandled exception", exc_info=(exc_type, exc_value, exc_traceback))
    # Keep the default exception handler
    sys.__excepthook__(exc_type, exc_value, exc_traceback)

sys.excepthook = handle_exception

# Fix for PyTorch and Streamlit compatibility issue
import warnings
warnings.filterwarnings('ignore', message='Examining the path of torch.classes')

# Fix for PyTorch classes path issue
import torch
torch.classes.__path__ = []

try:
    # Importing after warnings to avoid error messages
    logger.info("Importing MultiAgentSystem and video generation module")
    from agents.multi_agent_system import MultiAgentSystem
    from video_generation import generate_chat_video
except Exception as e:
    logger.error(f"Error importing modules: {str(e)}", exc_info=True)
    st.error(f"Error importing required modules: {str(e)}")
    
# Set page configuration
st.set_page_config(
    page_title="AI Content to Chat Video",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stTextArea textarea {
        height: 200px;
    }
    .agent-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .research-agent {
        background-color: #f0f7ff;
        border-left: 4px solid #0066cc;
    }
    .content-agent {
        background-color: #f0fff4;
        border-left: 4px solid #00cc66;
    }
    .chat-message {
        padding: 0.5rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
        display: inline-block;
        max-width: 80%;
    }
    .message-container {
        margin-bottom: 0.5rem;
        display: flex;
    }
    .message-left {
        justify-content: flex-start;
    }
    .message-right {
        justify-content: flex-end;
    }
    .message-left .chat-message {
        background-color: #e9e9eb;
        color: black;
    }
    .message-right .chat-message {
        background-color: #0b93f6;
        color: white;
    }
    .json-viewer {
        max-height: 300px;
        overflow-y: auto;
        background-color: #f6f6f6;
        padding: 1rem;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'agent_output' not in st.session_state:
    st.session_state.agent_output = None
if 'video_path' not in st.session_state:
    st.session_state.video_path = None
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'final_content' not in st.session_state:
    st.session_state.final_content = None
if 'is_processing' not in st.session_state:
    st.session_state.is_processing = False
if 'contact_name' not in st.session_state:
    st.session_state.contact_name = "Tracy"
if 'placeholder_text' not in st.session_state:
    st.session_state.placeholder_text = "Type a message..."
if 'speed' not in st.session_state:
    st.session_state.speed = 0.3
if 'current_query' not in st.session_state:
    st.session_state.current_query = ""
if 'generate_audio' not in st.session_state:
    st.session_state.generate_audio = False

def run_agent_system(query):
    # Ensure script run context is available for background operations
    get_or_create_script_run_ctx()
    
    st.session_state.is_processing = True
    st.session_state.current_query = query
    
    try:
        logger.info("Initializing multi-agent system")
        # Initialize the multi-agent system
        system = MultiAgentSystem()
        
        # Run the query
        with st.spinner("🧠 AI agents are processing your request..."):
            logger.info("Running query through multi-agent system")
            try:
                # Set a timeout for the query processing
                start_time = time.time()
                timeout = 120  # 2 minutes timeout
                
                # The workflow invocation might hang, so we'll monitor it
                result = None
                try:
                    result = system.run(query)
                    logger.info(f"Query processing completed in {time.time() - start_time:.2f} seconds")
                except Exception as e:
                    logger.error(f"Exception during system.run: {str(e)}", exc_info=True)
                    raise
                
                if result is None:
                    logger.error("system.run returned None")
                    raise ValueError("Failed to get result from AI system")
                
                st.session_state.agent_output = result
                
                if "messages" not in result:
                    logger.error(f"Missing 'messages' in result: {result}")
                    raise ValueError("Invalid result structure: missing 'messages'")
                
                st.session_state.messages = result["messages"]
                
                # Extract final content
                logger.info("Extracting final content")
                final_content = result.get("final_content")
                if final_content:
                    # Process the final content
                    if isinstance(final_content, str):
                        # Try to parse if it's JSON in a string
                        try:
                            # Remove the ```json and ``` markers if they exist
                            if final_content.startswith('```json'):
                                final_content = final_content[7:]
                            if final_content.endswith('```'):
                                final_content = final_content[:-3]
                            final_content = final_content.strip()
                            
                            # Parse JSON
                            logger.info("Parsing final content as JSON")
                            st.session_state.final_content = json.loads(final_content)
                        except json.JSONDecodeError as e:
                            logger.error(f"JSON parsing error: {str(e)}")
                            st.session_state.final_content = final_content
                    else:
                        st.session_state.final_content = final_content
                        
                    logger.info("Final content processed successfully")
                else:
                    logger.warning("No final content found in result")
                    st.warning("The AI system returned a response but no conversation content was generated. Please try again with a different query.")
            except Exception as e:
                logger.error(f"Error processing query: {str(e)}", exc_info=True)
                st.error(f"Error processing your request: {str(e)}")
                st.session_state.is_processing = False
                return
        
        # Generate video if we have content
        logger.info("Generating video")
        generate_video()
        
    except Exception as e:
        logger.error(f"Unexpected error in run_agent_system: {str(e)}", exc_info=True)
        st.error(f"An unexpected error occurred: {str(e)}")
    finally:
        st.session_state.is_processing = False
        logger.info("Processing completed")

def generate_video():
    """Generate video with current settings"""
    # Ensure script run context is available for background operations
    get_or_create_script_run_ctx()
    
    if st.session_state.final_content:
        with st.spinner("🎬 Generating chat video..."):
            # Create temporary directory for output
            if not os.path.exists("tmp"):
                os.makedirs("tmp")
            
            output_file = os.path.join("tmp", f"output_{int(time.time())}.mp4")
            
            # Save current state to allow recovery
            try:
                recovery_file = os.path.join("tmp", "last_state.json")
                with open(recovery_file, 'w') as f:
                    json.dump({
                        "final_content": st.session_state.final_content,
                        "contact_name": st.session_state.contact_name,
                        "placeholder_text": st.session_state.placeholder_text,
                        "speed": st.session_state.speed,
                        "generate_audio": st.session_state.generate_audio
                    }, f)
                logger.info("State saved for recovery")
            except Exception as e:
                logger.warning(f"Failed to save recovery state: {str(e)}")
                # Non-critical, just continue if saving state fails
                pass
                
            try:
                # Prevent torch runtime conflicts by ensuring a clean context
                import gc
                gc.collect()
                
                logger.info("Starting video generation with parameters: " + 
                           f"contact_name={st.session_state.contact_name}, " +
                           f"speed={st.session_state.speed}, " +
                           f"generate_audio={st.session_state.generate_audio}")
                
                output_path = generate_chat_video(
                    message_list=st.session_state.final_content,
                    placeholder_text=st.session_state.placeholder_text,
                    contact_name=st.session_state.contact_name,
                    output_file=output_file,
                    speed=st.session_state.speed,
                    generate_audio=st.session_state.generate_audio
                )
                st.session_state.video_path = output_path
                logger.info(f"Video generated successfully: {output_path}")
                return True
            except RuntimeError as e:
                if "no running event loop" in str(e) or "Tried to instantiate class" in str(e):
                    logger.error(f"PyTorch compatibility error: {str(e)}")
                    st.error("PyTorch compatibility error. Please try restarting the application.")
                    st.info("Workaround: Refresh the page and try with a smaller input or disable audio generation.")
                else:
                    logger.error(f"Error generating video: {str(e)}", exc_info=True)
                    st.error(f"Error generating video: {str(e)}")
                return False
            except Exception as e:
                logger.error(f"Error generating video: {str(e)}", exc_info=True)
                st.error(f"Error generating video: {str(e)}")
                return False
    else:
        logger.warning("Cannot generate video: No final content available")

def check_api_keys():
    """Check if required API keys are set"""
    # Load environment variables
    load_dotenv()
    
    # Check for Google API key (needed for the MultiAgentSystem)
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        st.sidebar.error("⚠️ Google API key not found. Please set the GOOGLE_API_KEY environment variable.")
        return False
    
    return True

# PyTorch compatibility fix
import asyncio
try:
    asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
except RuntimeError:
    # Already has an event loop
    pass

# Main UI
st.title("🤖 AI Content to Chat Video")

# Check API keys
api_keys_valid = check_api_keys()

# Check for recovery file on startup
try:
    recovery_file = os.path.join("tmp", "last_state.json")
    if os.path.exists(recovery_file):
        with st.sidebar.expander("🔄 Recovery Available", expanded=True):
            st.info("A previous session was interrupted. Do you want to recover?")
            if st.button("Recover Last Session"):
                with open(recovery_file, 'r') as f:
                    recovered_state = json.load(f)
                    st.session_state.final_content = recovered_state.get("final_content")
                    st.session_state.contact_name = recovered_state.get("contact_name", "Tracy")
                    st.session_state.placeholder_text = recovered_state.get("placeholder_text", "Type a message...")
                    st.session_state.speed = recovered_state.get("speed", 0.3)
                    st.session_state.generate_audio = recovered_state.get("generate_audio", False)
                    st.session_state.agent_output = {"messages": []}  # Dummy to show UI
                    
                    # Try to regenerate with recovered settings
                    generate_video()
                    st.experimental_rerun()
except Exception:
    # Non-critical, just continue if recovery fails
    pass

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    st.session_state.contact_name = st.text_input("Contact Name", value=st.session_state.contact_name)
    st.session_state.placeholder_text = st.text_input("Input Placeholder", value=st.session_state.placeholder_text)
    st.session_state.speed = st.slider("Video Speed", min_value=0.1, max_value=1.0, value=st.session_state.speed, step=0.1)
    st.session_state.generate_audio = st.checkbox("Generate Audio", value=st.session_state.generate_audio, 
                                                help="Disable this if you encounter PyTorch-related errors")
    
    st.markdown("---")
    
    # Add cache cleanup option
    if st.button("🧹 Clean Temporary Files"):
        try:
            import shutil
            if os.path.exists("tmp"):
                shutil.rmtree("tmp")
                os.makedirs("tmp")
            st.success("Temporary files cleaned successfully!")
        except Exception as e:
            st.error(f"Error cleaning temporary files: {str(e)}")
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This app uses a multi-agent system to:
    
    1. Research and analyze AI-related content
    2. Transform it into an engaging conversation
    3. Generate an animated chat video
    
    The content is processed by two specialized AI agents:
    - **Research Agent** gathers and analyzes information
    - **Content Creation Agent** transforms it into conversation
    """)

# Main content
st.markdown("""
📱 Enter AI-related news, research summaries, or critiques, and our system will transform it into an engaging chat conversation and animated video.
""")

# Input tabs
tab1, tab2 = st.tabs(["✏️ Text Input", "📚 Example Topics"])

with tab1:
    # Input area
    with st.form(key="input_form"):
        query = st.text_area(
            "Enter your AI-related news or research:",
            placeholder="Enter AI news, research summaries, or critiques here...",
            height=200
        )
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            submit_button = st.form_submit_button(
                "✨ Generate Content & Video",
                disabled=not api_keys_valid
            )
        
        if submit_button and query:
            if not query.strip():
                st.error("Please enter some content to process.")
            else:
                run_agent_system(query)

with tab2:
    # Example topics that users can select
    st.subheader("Select an example topic")
    
    example_topics = {
        "GPT-4 Overview": "GPT-4 is OpenAI's most advanced system, producing safer, more useful responses. It can solve difficult problems with greater accuracy, thanks to its broader general knowledge and problem-solving abilities. The system is optimized for chat but works well for traditional completions tasks. It performs at a human level on various professional and academic benchmarks.",
        "AI Research Ethics": "AI ethics addresses issues like privacy, bias, security, transparency, and accountability. It considers data collection ethics, algorithmic bias, social influence, security vulnerabilities, transparency in decision-making, and holding AI developers accountable. Major concerns include privacy risks, dataset biases, security threats, black-box problems, and accountability gaps.",
        "Diffusion Models": "Diffusion models are generative models that learn by gradually adding and then removing noise from data. They have excelled in image generation tasks like DALL-E, Midjourney, and Stable Diffusion. Unlike GANs, they can learn complex distributions without mode collapse and have shown success across various domains including image, audio, and video generation."
    }
    
    for topic_name, topic_content in example_topics.items():
        if st.button(f"📋 {topic_name}", key=f"example_{topic_name}"):
            run_agent_system(topic_content)

# Show loading indicator
if st.session_state.is_processing:
    st.info("⏳ Processing your request... Please wait.")

# Display results if available
if st.session_state.agent_output:
    st.subheader("🧠 Agent Outputs")
    
    # Display original query
    if st.session_state.current_query:
        st.markdown(f"""
        <div style="background-color: #f5f5f5; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;">
            <h4>Your Query</h4>
            <p>{st.session_state.current_query}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Create tabs for agent outputs
    output_tab1, output_tab2 = st.tabs(["💬 Processed Content", "🔍 Raw Agent Outputs"])
    
    with output_tab1:
        # Display final content
        if st.session_state.final_content:
            st.subheader("Conversation Content")
            
            if isinstance(st.session_state.final_content, list):
                # Display as chat messages
                chat_container = st.container()
                with chat_container:
                    for i, msg in enumerate(st.session_state.final_content):
                        is_right = i % 2 == 0  # Alternate left-right
                        align = "message-right" if is_right else "message-left"
                        
                        st.markdown(f"""
                        <div class="message-container {align}">
                            <div class="chat-message">
                                {msg}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                # If not a list, show as JSON
                st.markdown(f"""
                <div class="json-viewer">
                    <pre>{json.dumps(st.session_state.final_content, indent=2)}</pre>
                </div>
                """, unsafe_allow_html=True)
            
            # Video regeneration options
            st.subheader("Video Settings")
            col1, col2, col3 = st.columns(3)
            with col1:
                new_contact_name = st.text_input("Contact Name", value=st.session_state.contact_name, key="regen_name")
            with col2:
                new_placeholder = st.text_input("Input Placeholder", value=st.session_state.placeholder_text, key="regen_placeholder")
            with col3:
                new_speed = st.slider("Video Speed", min_value=0.1, max_value=1.0, value=st.session_state.speed, step=0.1, key="regen_speed")
            
            new_generate_audio = st.checkbox("Generate Audio", value=st.session_state.generate_audio, key="regen_audio",
                                          help="Disable this if you encounter PyTorch-related errors")
            
            if st.button("🔄 Regenerate Video with New Settings"):
                st.session_state.contact_name = new_contact_name
                st.session_state.placeholder_text = new_placeholder 
                st.session_state.speed = new_speed
                st.session_state.generate_audio = new_generate_audio
                with st.spinner("Regenerating video..."):
                    generate_video()
    
    with output_tab2:
        # Display agent messages
        for i, message in enumerate(st.session_state.messages):
            if i == 0:  # Skip the original query
                continue
                
            if "Research Agent:" in message:
                st.markdown(f"""
                <div class="agent-message research-agent">
                    <h4>Research Agent</h4>
                    {message.replace("Research Agent: ", "")}
                </div>
                """, unsafe_allow_html=True)
            elif "Content Agent:" in message:
                st.markdown(f"""
                <div class="agent-message content-agent">
                    <h4>Content Creation Agent</h4>
                    {message.replace("Content Agent: ", "")}
                </div>
                """, unsafe_allow_html=True)

    # Display video if available
    if st.session_state.video_path:
        st.subheader("🎬 Generated Chat Video")
        
        # Video container with custom styling
        video_container = st.container()
        with video_container:
            try:
                # Check if file exists
                if not os.path.exists(st.session_state.video_path):
                    st.error("Video file not found. It may have been deleted or moved.")
                else:
                    # Open the video file
                    video_file = open(st.session_state.video_path, 'rb')
                    video_bytes = video_file.read()
                    
                    # Create columns for the video and download button
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        # Display video
                        st.video(video_bytes)
                    with col2:
                        st.markdown("<br><br><br>", unsafe_allow_html=True)  # Add some vertical space
                        # Download button
                        st.download_button(
                            label="📥 Download Video",
                            data=video_bytes,
                            file_name="ai_conversation.mp4",
                            mime="video/mp4"
                        )
                        
                        # Only show info if we have list content
                        if isinstance(st.session_state.final_content, list):
                            st.markdown(f"""
                            <div style="background-color: #f0f7ff; padding: 10px; border-radius: 5px; margin-top: 10px;">
                                <p><strong>Video Info</strong></p>
                                <p>Duration: ~{len(st.session_state.final_content) * 2} seconds</p>
                                <p>Format: MP4</p>
                                <p>Resolution: 1080x1920</p>
                            </div>
                            """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error displaying video: {str(e)}")

# Footer with app info
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown(
        "<div style='text-align: center;'>"
        "Created with ❤️ using Streamlit<br>"
        "© 2023 AI Content to Chat Video"
        "</div>", 
        unsafe_allow_html=True
    ) 