from typing import TypedDict, Annotated, List
import operator
from dotenv import load_dotenv
import os
import logging
import time
import threading
import concurrent.futures
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[List[str], operator.add]
    research_output: str
    final_content: str

class MultiAgentSystem:
    def __init__(self):
        logger.info("Initializing MultiAgentSystem")
        try:
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                logger.error("GOOGLE_API_KEY environment variable not set")
                raise ValueError("GOOGLE_API_KEY environment variable not set")
                
            logger.info("Creating LLM client with API key")
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                google_api_key=api_key,
                temperature=0.7,
                timeout=60  # Add 60-second timeout for API calls
            )
            logger.info("Building workflow")
            self.workflow = self._build_workflow()
            logger.info("MultiAgentSystem initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing MultiAgentSystem: {str(e)}", exc_info=True)
            raise

    def invoke_with_timeout(self, llm, messages, timeout=60):
        """Invoke LLM with timeout to prevent hanging"""
        result = None
        error = None
        
        def invoke_llm():
            nonlocal result, error
            try:
                result = llm.invoke(messages)
            except Exception as e:
                error = e
        
        # Create and start thread
        thread = threading.Thread(target=invoke_llm)
        thread.daemon = True
        thread.start()
        
        # Wait for thread to complete with timeout
        thread.join(timeout)
        
        if thread.is_alive():
            # If still running after timeout, we have a problem
            logger.error(f"LLM call timed out after {timeout} seconds")
            raise TimeoutError(f"LLM call timed out after {timeout} seconds")
        
        if error:
            raise error
            
        return result

    def research_agent(self, state: AgentState) -> AgentState:
        logger.info("Research agent starting")
        start_time = time.time()
        try:
            prompt = SystemMessage(content="You are a research assistant. Gather concise information based on the user's input for transforming AI-related news, research summaries, or critiques into engaging, conversational content for a general audience.")
            user_input = state["messages"][-1]
            logger.info(f"Invoking LLM with research prompt - Input length: {len(user_input)}")
            
            # Use timeout-protected invocation
            response = self.invoke_with_timeout(
                self.llm, 
                [prompt, HumanMessage(content=user_input)],
                timeout=60
            )
            
            logger.info(f"Research agent completed in {time.time() - start_time:.2f} seconds")
            state["research_output"] = response.content
            state["messages"] = state["messages"] + [f"Research Agent: {response.content}"]
            return state
        except TimeoutError as e:
            logger.error(f"Timeout in research agent: {str(e)}")
            state["research_output"] = "Error: The research process timed out. Please try again with a shorter or simpler query."
            state["messages"] = state["messages"] + [f"Research Agent: Error: {str(e)}"]
            return state
        except Exception as e:
            logger.error(f"Error in research agent: {str(e)}", exc_info=True)
            # Return state with error message instead of failing
            state["research_output"] = f"Error in research process: {str(e)}"
            state["messages"] = state["messages"] + [f"Research Agent: Error: {str(e)}"]
            return state

    def content_creation_agent(self, state: AgentState) -> AgentState:
        logger.info("Content creation agent starting")
        start_time = time.time()
        try:
            prompt = SystemMessage(content="""

                You are a creative content writer tasked with transforming AI-related news, research summaries, or critiques into engaging, conversational content for a general audience. You will be provided with a piece of text related to AI, and your job is to create a conversation between two people that makes the topic accessible, interesting, and easy to understand.

                ### Instructions:

                - **Tone and Style**:  
                Craft the conversation in a casual, texting style, using emojis to make it feel like a real chat between friends. Since it will be animated as a texting chat, keep segments concise and impactful.

                - **Content**:  
                Explain, discuss, or critique the provided topic in a simple, engaging way. Cover the main ideas from the input text, simplifying complex concepts without losing their core meaning.

                - **Structure**:  
                Generate at least 4 segments, with the option to add more if needed to fully explore the topic. Each segment should be a complete, self-contained message of about 10 words.

                - **Approach**:  
                Have one person ask curious or skeptical questions while the other explains, or let both discuss from different perspectives. Add humor, curiosity, or skepticism for liveliness. Get creative with the characters' personalities or expressions to boost engagement.

                - **Audience**:  
                Target a general audience, prioritizing clarity, simplicity, and relatability over technical depth. Avoid jargon unless essential, and explain it simply if used.

                - **Flow**:  
                Ensure the conversation flows naturally, with each segment building on the previous one for a coherent dialogue.

                ### Output Format:

                Return your response as a JSON array of strings, where each string is a segment of the conversation.

                ---

                This prompt delivers dynamic, animation-ready conversations that demystify AI topics with flair and accessibility!
            """)
            research_data = state["research_output"]
            logger.info(f"Invoking LLM with content creation prompt - Research data length: {len(research_data)}")
            
            # Use timeout-protected invocation
            response = self.invoke_with_timeout(
                self.llm, 
                [prompt, HumanMessage(content=f"Based on this research: {research_data}")],
                timeout=60
            )
            
            logger.info(f"Content creation agent completed in {time.time() - start_time:.2f} seconds")
            state["final_content"] = response.content
            state["messages"] = state["messages"] + [f"Content Agent: {response.content}"]
            return state
        except TimeoutError as e:
            logger.error(f"Timeout in content creation agent: {str(e)}")
            state["final_content"] = "Error: The content creation process timed out. Please try again with simpler research data."
            state["messages"] = state["messages"] + [f"Content Agent: Error: {str(e)}"]
            return state
        except Exception as e:
            logger.error(f"Error in content creation agent: {str(e)}", exc_info=True)
            # Return state with error message instead of failing
            state["final_content"] = f"Error in content creation: {str(e)}"
            state["messages"] = state["messages"] + [f"Content Agent: Error: {str(e)}"]
            return state

    def _build_workflow(self):
        logger.info("Building workflow graph")
        try:
            workflow = StateGraph(AgentState)
            workflow.add_node("research", self.research_agent)
            workflow.add_node("content", self.content_creation_agent)
            workflow.set_entry_point("research")
            workflow.add_edge("research", "content")
            workflow.add_edge("content", END)
            logger.info("Compiling workflow")
            return workflow.compile()
        except Exception as e:
            logger.error(f"Error building workflow: {str(e)}", exc_info=True)
            raise

    def run(self, query: str):
        logger.info(f"Running query through MultiAgentSystem - Query length: {len(query)}")
        start_time = time.time()
        try:
            initial_state = {
                "messages": [query],
                "research_output": "",
                "final_content": ""
            }
            
            logger.info("Invoking workflow")
            result = self.workflow.invoke(initial_state)
            logger.info(f"Workflow completed in {time.time() - start_time:.2f} seconds")
            return result
        except Exception as e:
            logger.error(f"Error in workflow execution: {str(e)}", exc_info=True)
            # Return a basic result structure with the error
            return {
                "messages": [query, f"Error: {str(e)}"],
                "research_output": "",
                "final_content": f"Error occurred: {str(e)}"
            } 