from crewai import Agent, Task, Crew, LLM
from crewai_tools import SerperDevTool
import streamlit as st
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

st.set_page_config(page_title="Content Researcher & Writer", page_icon="", layout="wide")

st.title("Content Researcher & Writer, powered by CrewAI")
st.markdown("Generate blog post about any topic using AI agents.")

# Sidebar
with st.sidebar:
    st.header("Content Settings")
    topic = st.text_area(
        "Enter topic for the blog post",
        height=100,
        placeholder="Enter the topic"
    )
    
    st.markdown("LLM Settings")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7)
    
    # Add API key input
    api_key = st.text_input("Enter Gemini API Key", type="password")
    
    st.markdown("___")
    generate_button = st.button("Generate Content", type="primary", use_container_width=True)

def generate_content(topic, api_key, temperature):
    try:
        # Initialize the LLM with error handling
        if not api_key:
            raise ValueError("API key is required")
            
        llm = LLM(
            model="gemini/gemini-1.5-pro-latest",
            temperature=temperature,
            api_key=api_key
        )
        
        # Initialize search tool with error handling
        try:
            search_tool = SerperDevTool(n=10)
        except Exception as e:
            st.warning("Search tool initialization failed. Proceeding without search capability.")
            search_tool = None
        
        # Define agents with proper error handling
        tools = [search_tool] if search_tool else []
        
        senior_research_analyst = Agent(
            role="Senior Research Analyst",
            goal=f"Research, analyze, and synthesize comprehensive information on {topic}",
            backstory="You're an expert research analyst with advanced research skills.",
            allow_delegation=False,
            verbose=True,
            tools=tools,
            llm=llm
        )
        
        content_writer = Agent(
            role="Content Writer",
            goal="Transform research findings into engaging blog posts while maintaining accuracy",
            backstory="You're a skilled content writer specialized in creating engaging content.",
            allow_delegation=False,
            tools=tools,
            llm=llm
        )
        
        # Define tasks with proper validation
        if not topic:
            raise ValueError("Topic is required")
            
        research_task = Task(
            description=f"Research {topic} comprehensively including recent developments and trends.",
            expected_output="A detailed research report with key findings and analysis.",
            agent=senior_research_analyst
        )
        
        writing_task = Task(
            description="Create an engaging blog post from the research findings.",
            expected_output="A polished blog post in markdown format.",
            agent=content_writer
        )
        
        # Create and execute crew
        crew = Crew(
            agents=[senior_research_analyst, content_writer],
            tasks=[research_task, writing_task],
            verbose=True
        )
        
        # Execute with timeout and error handling
        with st.spinner('Processing...'):
            result = crew.kickoff(inputs={"topic": topic})
            if not result:
                raise ValueError("No content generated")
            return result
            
    except Exception as e:
        raise Exception(f"Content generation failed: {str(e)}")

if generate_button:
    if not topic:
        st.error("Please enter a topic")
    else:
        try:
            result = generate_content(topic, api_key, temperature)
            st.markdown("### Generated Content")
            st.markdown(result)
            
            # Add download button with error handling
            try:
                st.download_button(
                    label="Download Content",
                    data=result,
                    file_name=f"{topic.lower().replace(' ', '_')}_article.md",
                    mime="text/markdown"
                )
            except Exception as e:
                st.error(f"Failed to create download button: {str(e)}")
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

st.markdown("___")
st.markdown("Built with CrewAI, Streamlit and Gemini")