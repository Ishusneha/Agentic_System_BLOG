from crewai import Agent, Task,Crew,LLM
from crewai_tools import SerperDevTool
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(page_title="Content Researcher & Writer",page_icon="",layout="wide")

st.title("Content Researcher & Writer, powered by CrewAI")
st.markdown("Generate blog post about any topic using AI agents.")

#Sidebar
with st.sidebar:
    st.header("Content Settings")

    #Make the text input take up more space
    topic=st.text_area(
        "Enter topic for the blog post",
        height=100,
        placeholder="Enter the topic"

    )

    st.markdown("LLM Settings")
    temperature=st.slider("Temperature",0.0,1.0,0.7)

    st.markdown("___")

    generate_button=st.button("Generate Content",type="primary",use_container_width=True)

    with st.expander("How to use"):
        st.markdown("""
                    1.Enter your desired content topic
                    2.Play with the temperature
                    3. CLick 'Generate Content' to start
                    4. Wait for the AI to generate your article
                    5. Download the result as a markdown file""")
def generate_content(topic):
    # Initialize the LLM model
    llm = LLM(
        model="gemini/gemini-1.5-pro-latest",
        temperature=0.7,
        api_key="GEMINI_API_KEY"
    )

    # Initialize the web search tool
    search_tool = SerperDevTool(n=10)

    # Define Agent 1: Senior Research Analyst
    senior_research_analyst = Agent(
        role="Senior Research Analyst",
        goal=f"Research, analyze, and synthesize comprehensive information on {topic} from reliable web sources",
        backstory=(
            "You're an expert research analyst with advanced web research skills. "
            "You excel at finding, analyzing, and synthesizing information from "
            "across the internet using search tools. You're skilled at "
            "fact-checking, cross-referencing information, and "
            "identifying key patterns and insights. You provide "
            "well-organized research briefs with proper citations "
            "and source verification. Your analysis includes both "
            "raw data and interpreted insights, making complex "
            "information accessible and actionable."
        ),
        allow_delegation=False,  # This agent won't delegate tasks to others
        verbose=True,
        tools=[search_tool],
        llm=llm
    )

    # Define Agent 2: Content Writer
    content_writer = Agent(
        role="Content Writer",
        goal="Transform research findings into engaging blog posts while maintaining accuracy",
        backstory=(
            "You're a skilled content writer specialized in creating "
            "engaging, accessible content from technical research. "
            "You work closely with the Senior Research Analyst and excel at maintaining the perfect "
            "balance between accuracy and readability. You're skilled at "
            "identifying key takeaways, crafting compelling headlines, and writing clear, concise content "
            "that resonates with the target audience. Your writing is well-structured and easy to understand."
        ),
        allow_delegation=False,
        tools=[search_tool],
        llm=llm
    )


    # Define Task 1: Research Task
    research_task = Task(
        description=(
            f"""
            1. Conduct comprehensive research on {topic}, including:
                - Recent developments and news
                - Key industry trends and innovations
                - Expert opinions and analyses
                - Statistical data and market insights
            2. Evaluate source credibility and fact-check all information
            3. Organize findings into a structured research brief
            4. Include all relevant citations and sources
            """
        ),
        expected_output=(
            """
            A detailed research report containing:
            - Executive summary of key findings
            - Comprehensive analysis of current trends and developments
            - List of verified facts and statistics
            - All citations and links to original sources
            - Clear categorization of main themes and patterns
            Please format with clear sections and bullet points for easy reference.
            """
        ),
        agent=senior_research_analyst
    )

    # Define Task 2: Content Writing Task
    writing_task = Task(
        description=(
            """
            Using the research brief provided, create an engaging blog post that:
            1. Transforms technical information into accessible content
            2. Maintains all factual accuracy and citations from the research
            3. Includes:
                - Attention-grabbing introduction
                - Well-structured body sections with clear headings
                - Compelling conclusion
            4. Preserves all source citations in [Source:URL] format
            5. Includes a References section at the end
            """
        ),
        expected_output=(
            """
            A polished blog post in markdown format that:
            - Engages readers while maintaining accuracy
            - Contains properly structured sections
            - Includes inline citations hyperlinked to the original source URL
            - Presents information in an accessible yet informative way
            - Follows proper markdown formatting, using H1 for the title and H3 for the sub-sections
            """
        ),
        agent=content_writer
    )

    # Define Crew to execute tasks
    crew = Crew(
        agents=[senior_research_analyst, content_writer],
        tasks=[research_task, writing_task],
        verbose=True
    )

    # Execute the tasks
    result = crew.kickoff(inputs={"topic": topic})
    print(result)
    return result

if generate_button:
    with st.spinner('Generating content... This may take a moment'):
        try:
            result=generate_content(topic)
            st.markdown("Generated Content")
            st.markdown(result)

            st.download_button(
                label="Download Content",
                data=result.raw,
                file_name=f"{topic.lower().replace('','_')}_article.md",
                mime="text/markdown",

            )
        except Exception as e:
            st.error(f"An error occured:{str(e)}")
st.markdown("___")
st.markdown("Build with CrewAI, Streamlit and Gemini")
