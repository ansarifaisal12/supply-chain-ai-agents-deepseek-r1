import streamlit as st
import pandas as pd
import numpy as np
from langchain.chains import LLMChain, RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.llms import Ollama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import DataFrameLoader
import os
import mlflow
import mlflow.sklearn
from datetime import datetime

# Custom Streamlit theme
st.set_page_config(
    page_title="Multi-AI Agents with DeepSeek-r1",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "## Multi-AI Agents for Supply Chain Optimization\nPowered by DeepSeek-r1"
    }
)

# Custom CSS for styling
st.markdown("""
    <style>
    .stApp {
        background-color: #f5f5f5;
        color: #333333;
    }
    .stSidebar {
        background-color: #2c3e50 !important;
        color: #ffffff !important;
    }
    .stButton button {
        background-color: #3498db;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 16px;
    }
    .stButton button:hover {
        background-color: #2980b9;
    }
    .stTextInput input {
        border-radius: 5px;
        padding: 10px;
    }
    .stHeader {
        font-size: 2.5em;
        font-weight: bold;
        color: #2c3e50;
    }
    .stSubheader {
        font-size: 1.5em;
        font-weight: bold;
        color: #3498db;
    }
    .stMarkdown {
        font-size: 1.1em;
        line-height: 1.6;
    }
    .stSuccess {
        color: #27ae60;
    }
    .stError {
        color: #e74c3c;
    }
    </style>
    """, unsafe_allow_html=True)

# Load sample data
def load_data(folder_path='sample_data'):
    inventory_data = pd.read_csv(f'{folder_path}/sample_inventory_data.csv')
    sales_data = pd.read_csv(f'{folder_path}/sample_sales_data.csv')
    supplier_data = pd.read_csv(f'{folder_path}/sample_supplier_data.csv')
    return inventory_data, sales_data, supplier_data

# Initialize embedding model
def initialize_embedding_model():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Create vector store from data
def create_vector_store(data, embeddings, column_name):
    loader = DataFrameLoader(data, page_content_column=column_name)
    documents = loader.load()
    
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    
    vector_store = FAISS.from_documents(texts, embeddings)
    return vector_store

# Initialize AI agents with enhanced prompts and context
def initialize_ai_agents(inventory_vector_store, sales_vector_store, supplier_vector_store):
    ollama_llm = Ollama(model="deepseek-r1:7b")
    
    # Enhanced Inventory Management Agent
    inventory_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
        You are an inventory management expert. Use the following context to provide recommendations for reorder points, stockouts, and inventory optimization. Consider the following factors:
        - Current stock levels
        - Historical sales data
        - Lead times
        - Seasonal trends
        - Supplier reliability

        Context: {context}
        Question: {question}

        Provide a detailed analysis and actionable recommendations.
        """
    )
    inventory_agent = RetrievalQA.from_chain_type(
        llm=ollama_llm,
        chain_type="stuff",
        retriever=inventory_vector_store.as_retriever(),
        chain_type_kwargs={"prompt": inventory_prompt}
    )
    
    # Enhanced Sales Forecasting Agent
    sales_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
        You are a sales forecasting expert. Use the following context to predict future sales trends and provide actionable insights. Consider the following factors:
        - Historical sales data
        - Seasonal trends
        - Market conditions
        - Product lifecycle
        - Promotional activities

        Context: {context}
        Question: {question}

        Provide a detailed forecast and actionable recommendations.
        """
    )
    sales_agent = RetrievalQA.from_chain_type(
        llm=ollama_llm,
        chain_type="stuff",
        retriever=sales_vector_store.as_retriever(),
        chain_type_kwargs={"prompt": sales_prompt}
    )
    
    # Enhanced Supplier Evaluation Agent
    supplier_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
        You are a supplier evaluation expert. Use the following context to evaluate supplier performance and suggest the best suppliers. Consider the following factors:
        - Lead times
        - Cost markup
        - Reliability score
        - Historical performance
        - Market reputation

        Context: {context}
        Question: {question}

        Provide a detailed evaluation and actionable recommendations.
        """
    )
    supplier_agent = RetrievalQA.from_chain_type(
        llm=ollama_llm,
        chain_type="stuff",
        retriever=supplier_vector_store.as_retriever(),
        chain_type_kwargs={"prompt": supplier_prompt}
    )
    
    return inventory_agent, sales_agent, supplier_agent

# Streamlit app
def main():
    # Sidebar with logo and navigation
    st.sidebar.markdown("## Navigation")
    task = st.sidebar.radio("Select Task", ["Inventory Management", "Sales Forecasting", "Supplier Evaluation"])
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.markdown("This application uses AI agents to optimize supply chain operations. Select a task to get started.")

    # Main content
    st.markdown("<div class='stHeader'>🐋 Multi-AI Agents with DeepSeek-r1</div>", unsafe_allow_html=True)
    st.markdown("---")

    # Load data
    inventory_data, sales_data, supplier_data = load_data()
    
    # Initialize embedding model
    embeddings = initialize_embedding_model()
    
    # Create vector stores for each dataset
    inventory_vector_store = create_vector_store(inventory_data, embeddings, "product_name")
    sales_vector_store = create_vector_store(sales_data, embeddings, "product_name")
    supplier_vector_store = create_vector_store(supplier_data, embeddings, "supplier_name")
    
    # Initialize AI agents
    inventory_agent, sales_agent, supplier_agent = initialize_ai_agents(inventory_vector_store, sales_vector_store, supplier_vector_store)
    
    # Initialize MLflow
    mlflow.set_tracking_uri("http://127.0.0.1:5000")  # Set your MLflow tracking URI
    mlflow.set_experiment("Supply_Chain_Optimization")

    # Task-specific UI
    if task == "Inventory Management":
        st.markdown("<div class='stSubheader'>📦 Inventory Management Agent</div>", unsafe_allow_html=True)
        query = st.text_input("Enter your query (e.g., 'Suggest reorder points for Product_1'):")
        if query:
            with st.spinner("Analyzing inventory data..."):
                try:
                    start_time = datetime.now()
                    response = inventory_agent.run(query)
                    end_time = datetime.now()
                    duration = (end_time - start_time).total_seconds()
                    
                    # Log metrics to MLflow
                    with mlflow.start_run():
                        mlflow.log_metric("response_time", duration)
                        mlflow.log_param("query", query)
                        mlflow.log_text(response, "response.txt")
                    
                    st.markdown("<div class='stSubheader'>📝 Response:</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='stMarkdown'>{response}</div>", unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
    
    elif task == "Sales Forecasting":
        st.markdown("<div class='stSubheader'>📈 Sales Forecasting Agent</div>", unsafe_allow_html=True)
        query = st.text_input("Enter your query (e.g., 'Predict sales for Product_2 in the next quarter'):")
        if query:
            with st.spinner("Analyzing sales data..."):
                try:
                    start_time = datetime.now()
                    response = sales_agent.run(query)
                    end_time = datetime.now()
                    duration = (end_time - start_time).total_seconds()
                    
                    # Log metrics to MLflow
                    with mlflow.start_run():
                        mlflow.log_metric("response_time", duration)
                        mlflow.log_param("query", query)
                        mlflow.log_text(response, "response.txt")
                    
                    st.markdown("<div class='stSubheader'>📝 Response:</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='stMarkdown'>{response}</div>", unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
    
    elif task == "Supplier Evaluation":
        st.markdown("<div class='stSubheader'>🏭 Supplier Evaluation Agent</div>", unsafe_allow_html=True)
        query = st.text_input("Enter your query (e.g., 'Evaluate Supplier_1 performance'):")
        if query:
            with st.spinner("Analyzing supplier data..."):
                try:
                    start_time = datetime.now()
                    response = supplier_agent.run(query)
                    end_time = datetime.now()
                    duration = (end_time - start_time).total_seconds()
                    
                    # Log metrics to MLflow
                    with mlflow.start_run():
                        mlflow.log_metric("response_time", duration)
                        mlflow.log_param("query", query)
                        mlflow.log_text(response, "response.txt")
                    
                    st.markdown("<div class='stSubheader'>📝 Response:</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='stMarkdown'>{response}</div>", unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()