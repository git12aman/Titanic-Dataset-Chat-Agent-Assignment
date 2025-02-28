import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
# Updated import statement
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_types import AgentType
import io
import base64
import os
import json
import uvicorn
from threading import Thread
import numpy as np

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the Titanic dataset
@st.cache_data
def load_data():
    # In a real implementation, you would load from a CSV file
    # For this example, we'll create a simplified dataset
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    try:
        df = pd.read_csv(url)
        return df
    except:
        # Fallback to a small sample if the URL is unavailable
        data = {
            'PassengerId': range(1, 11),
            'Survived': [0, 1, 1, 1, 0, 0, 0, 1, 1, 0],
            'Pclass': [3, 1, 3, 1, 3, 3, 1, 3, 2, 3],
            'Name': ['Braund, Mr. Owen Harris', 'Cumings, Mrs. John Bradley', 'Heikkinen, Miss. Laina', 
                    'Futrelle, Mrs. Jacques Heath', 'Allen, Mr. William Henry', 'Moran, Mr. James', 
                    'McCarthy, Mr. Timothy J', 'Palsson, Master. Gosta Leonard', 'Johnson, Mrs. Oscar W', 
                    'Nasser, Mrs. Nicholas'],
            'Sex': ['male', 'female', 'female', 'female', 'male', 'male', 'male', 'male', 'female', 'female'],
            'Age': [22, 38, 26, 35, 35, None, 54, 2, 27, 14],
            'SibSp': [1, 1, 0, 1, 0, 0, 0, 3, 0, 1],
            'Parch': [0, 0, 0, 0, 0, 0, 0, 1, 2, 0],
            'Ticket': ['A/5 21171', 'PC 17599', 'STON/O2. 3101282', '113803', '373450', '330877', 
                      '17463', '349909', '347742', '237736'],
            'Fare': [7.25, 71.2833, 7.925, 53.1, 8.05, 8.4583, 51.8625, 21.075, 11.1333, 30.0708],
            'Cabin': [None, 'C85', None, 'C123', None, None, 'E46', None, None, None],
            'Embarked': ['S', 'C', 'S', 'S', 'S', 'Q', 'S', 'S', 'S', 'C']
        }
        return pd.DataFrame(data)
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)
from dotenv import load_dotenv
load_dotenv()

import os

os.environ["OPENAI_API_KEY"] = "AIzaSyAqy08ixa1Uxv5iBuPNUCDg8jbi03J696M"

environ = "AIzaSyAqy08ixa1Uxv5iBuPNUCDg8jbi03J696M"


if not environ:
    raise ValueError("Google API Key is missing! Set it as an environment variable.")

# Initialize the LangChain agent
def get_agent():
    df = load_data()
    
    # Make sure to set your OpenAI API key in the environment variables
    #os.environ["OPENAI_API_KEY"] = "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"
    
    return create_pandas_dataframe_agent(
    ChatOpenAI(temperature=0, model="gpt-3.5-turbo"),
    df,
    verbose=True,
    agent_type=AgentType.OPENAI_FUNCTIONS,
    allow_dangerous_code=True  # Explicitly allowing execution of code
)


# Function to generate visualizations
def generate_visualization(query: str):
    df = load_data()
    plt.figure(figsize=(10, 6))
    
    if "histogram" in query.lower() and "age" in query.lower():
        sns.histplot(df['Age'].dropna(), kde=True)
        plt.title('Histogram of Passenger Ages')
        plt.xlabel('Age')
        plt.ylabel('Count')
    
    elif "survival" in query.lower() and ("gender" in query.lower() or "sex" in query.lower() or "male" in query.lower() or "female" in query.lower()):
        survival_by_sex = df.groupby('Sex')['Survived'].mean() * 100
        ax = survival_by_sex.plot(kind='bar', color=['#ff9999', '#66b3ff'])
        plt.title('Survival Rate by Gender')
        plt.xlabel('Gender')
        plt.ylabel('Survival Rate (%)')
        plt.xticks(rotation=0)
        
        # Add percentage labels on bars
        for i, v in enumerate(survival_by_sex):
            ax.text(i, v + 1, f"{v:.1f}%", ha='center')
    
    elif "class" in query.lower() and "survival" in query.lower():
        survival_by_class = df.groupby('Pclass')['Survived'].mean() * 100
        ax = survival_by_class.plot(kind='bar', color=['#ff9999', '#66b3ff', '#99ff99'])
        plt.title('Survival Rate by Passenger Class')
        plt.xlabel('Class (1 = 1st, 2 = 2nd, 3 = 3rd)')
        plt.ylabel('Survival Rate (%)')
        plt.xticks(rotation=0)
        
        # Add percentage labels on bars
        for i, v in enumerate(survival_by_class):
            ax.text(i, v + 1, f"{v:.1f}%", ha='center')
    
    elif "embark" in query.lower() or "port" in query.lower():
        # Replace NaN values with 'Unknown'
        df['Embarked'] = df['Embarked'].fillna('Unknown')
        
        # Create a mapping for port names
        port_names = {'C': 'Cherbourg', 'Q': 'Queenstown', 'S': 'Southampton', 'Unknown': 'Unknown'}
        
        # Count passengers by embarkation port
        embarked_counts = df['Embarked'].value_counts()
        
        # Create a new Series with full port names
        embarked_counts_named = pd.Series({
            port_names.get(port, port): count 
            for port, count in embarked_counts.items()
        })
        
        # Plot
        ax = embarked_counts_named.plot(kind='bar', color=['#ff9999', '#66b3ff', '#99ff99'])
        plt.title('Passengers by Embarkation Port')
        plt.xlabel('Port')
        plt.ylabel('Number of Passengers')
        plt.xticks(rotation=45)
        
        # Add count labels on bars
        for i, v in enumerate(embarked_counts_named):
            ax.text(i, v + 5, str(v), ha='center')
    
    elif "fare" in query.lower() or "ticket" in query.lower() or "price" in query.lower():
        if "average" in query.lower() or "mean" in query.lower():
            # Calculate average fare by class
            avg_fare_by_class = df.groupby('Pclass')['Fare'].mean()
            ax = avg_fare_by_class.plot(kind='bar', color=['#ff9999', '#66b3ff', '#99ff99'])
            plt.title('Average Ticket Fare by Passenger Class')
            plt.xlabel('Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)')
            plt.ylabel('Average Fare ($)')
            plt.xticks(rotation=0)
            
            # Add fare labels on bars
            for i, v in enumerate(avg_fare_by_class):
                ax.text(i, v + 1, f"${v:.2f}", ha='center')
        else:
            # Box plot of fares by class
            sns.boxplot(x='Pclass', y='Fare', data=df)
            plt.title('Ticket Fare Distribution by Passenger Class')
            plt.xlabel('Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)')
            plt.ylabel('Fare ($)')
    
    elif "percentage" in query.lower() and "survived" in query.lower():
        # Calculate overall survival rate
        survival_rate = df['Survived'].mean() * 100
        
        # Create a pie chart
        labels = ['Did not survive', 'Survived']
        sizes = [(100 - survival_rate), survival_rate]
        colors = ['#ff9999', '#66b3ff']
        explode = (0, 0.1)  # explode the 2nd slice (Survived)
        
        plt.pie(sizes, explode=explode, labels=labels, colors=colors,
                autopct='%1.1f%%', shadow=True, startangle=90)
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        plt.title('Passenger Survival Rate')
    
    else:
        # Default visualization - survival count
        survived_counts = df['Survived'].value_counts()
        ax = survived_counts.plot(kind='bar', color=['#ff9999', '#66b3ff'])
        plt.title('Survival Count')
        plt.xlabel('Survived (0 = No, 1 = Yes)')
        plt.ylabel('Number of Passengers')
        plt.xticks(rotation=0, ticks=[0, 1], labels=['Did not survive', 'Survived'])
        
        # Add count labels on bars
        for i, v in enumerate(survived_counts):
            ax.text(i, v + 5, str(v), ha='center')
    
    # Save the plot to a bytes buffer
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    
    # Encode the image to base64
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return f"data:image/png;base64,{img_str}"

# API endpoint for chat
@app.post("/api/chat")
async def chat(request: Request):
    data = await request.json()
    query = data.get("query", "")
    
    if not query:
        return {"response": "Please provide a query about the Titanic dataset."}
    
    try:
        agent = get_agent()
        response = agent.run(query)
        
        # Check if visualization is needed
        needs_viz = any(keyword in query.lower() for keyword in 
                      ["show", "plot", "graph", "visualize", "histogram", "chart", "percentage", "distribution"])
        
        viz_url = None
        if needs_viz:
            viz_url = generate_visualization(query)
        
        return {"response": response, "visualization": viz_url}
    except Exception as e:
        return {"response": f"Error processing your query: {str(e)}", "visualization": None}

# API endpoint for visualizations
@app.get("/api/visualization")
async def visualization(query: str):
    try:
        viz_url = generate_visualization(query)
        return {"url": viz_url}
    except Exception as e:
        return {"error": str(e)}

# Streamlit UI
def streamlit_app():
    st.set_page_config(
        page_title="Titanic Dataset Analyzer", 
        page_icon="🚢",
        layout="wide"
    )
    
    st.title("🚢 Titanic Dataset Analyzer")
    st.write("Ask questions about the Titanic dataset and get both text answers and visual insights!")
    
    # Sidebar with example questions
    st.sidebar.title("Example Questions")
    example_questions = [
        "What percentage of passengers survived on the Titanic?",
        "Show me a histogram of passenger ages",
        "What was the average ticket fare?",
        "How many passengers embarked from each port?",
        "Compare survival rates between males and females",
        "What class had the highest survival rate?"
    ]
    
    # Create buttons for example questions
    for question in example_questions:
        if st.sidebar.button(question):
            st.session_state.user_input = question
            st.session_state.submit_clicked = True
    
    # Display dataset info in the sidebar
    with st.sidebar.expander("Dataset Information"):
        st.write("""
        The Titanic dataset contains information about passengers aboard the RMS Titanic, which sank on April 15, 1912.
        
        **Columns:**
        - **PassengerId**: Unique identifier for each passenger
        - **Survived**: Whether the passenger survived (1) or not (0)
        - **Pclass**: Passenger class (1 = 1st, 2 = 2nd, 3 = 3rd)
        - **Name**: Passenger name
        - **Sex**: Passenger gender
        - **Age**: Passenger age
        - **SibSp**: Number of siblings/spouses aboard
        - **Parch**: Number of parents/children aboard
        - **Ticket**: Ticket number
        - **Fare**: Passenger fare
        - **Cabin**: Cabin number
        - **Embarked**: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)
        """)
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "user_input" not in st.session_state:
        st.session_state.user_input = ""
    
    if "submit_clicked" not in st.session_state:
        st.session_state.submit_clicked = False
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "visualization" in message and message["visualization"]:
                st.image(message["visualization"])
    
    # Chat input
    user_input = st.chat_input("Ask about the Titanic dataset...", key="chat_input")
    
    # Process input when submitted via chat input
    if user_input:
        st.session_state.user_input = user_input
        st.session_state.submit_clicked = True
    
    # Process the query when submit is clicked
    if st.session_state.submit_clicked:
        query = st.session_state.user_input
        
        # Reset the submit flag
        st.session_state.submit_clicked = False
        
        if query:
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": query})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(query)
            
            # Get response from agent
            with st.chat_message("assistant"):
                with st.spinner("Analyzing the Titanic dataset..."):
                    try:
                        # In a real implementation, you would call your FastAPI endpoint
                        # For simplicity, we'll call the functions directly
                        agent = get_agent()
                        response = agent.run(query)
                        
                        # Check if visualization is needed
                        needs_viz = any(keyword in query.lower() for keyword in 
                                      ["show", "plot", "graph", "visualize", "histogram", "chart", 
                                       "percentage", "distribution", "compare"])
                        
                        st.markdown(response)
                        
                        visualization_url = None
                        if needs_viz:
                            visualization_url = generate_visualization(query)
                            st.image(visualization_url)
                        
                        # Add assistant response to chat history
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": response,
                            "visualization": visualization_url
                        })
                    except Exception as e:
                        error_message = f"Error analyzing the data: {str(e)}"
                        st.error(error_message)
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": error_message
                        })

# Function to run FastAPI in a separate thread
def run_fastapi():
    uvicorn.run(app, host="0.0.0.0", port=8000)

# Main function to run both Streamlit and FastAPI
def main():
    # Start FastAPI in a separate thread
    api_thread = Thread(target=run_fastapi)
    api_thread.daemon = True
    api_thread.start()
    
    # Run Streamlit app
    streamlit_app()

if __name__ == "__main__":
    # For deployment, you might want to run these separately
    # For local development, this runs both together
    main()