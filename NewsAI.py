#!/usr/bin/env python
# coding: utf-8

# setting up the environment, importing libraries and frameworks

import streamlit as st
import numpy as np
import os
import random
import json
from newsapi import NewsApiClient
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.tools import DuckDuckGoSearchResults

from groq import Groq

#Initialization
#Empty list to store news articles for processing
filtered_articles = []

# Define sources and API parameters
australian_news_sources = [
    "abc-news-au", "news-com-au", "australian-financial-review", "google-news-au",
    "bbc-news", "bbc-sport", "bloomberg", "business-insider", "cbc-news", "cnn",
    "espn", "espn-cric-info", "financial-post", "fox-news", "fox-sports",
    "the-washington-post", "the-wall-street-journal", "the-huffington-post", "reuters"
]
sources_string = ','.join(australian_news_sources)

#API Key for NewsAPI
api_key = 'bd60dd75f6ca43709e8835107e8758d9'

# News Article Category keywords
category_keywords = {
    'sports': ['sports', 'football', 'basketball', 'f1', 'soccer', 'cricket', 'rugby', 'tennis'],
    'lifestyle': ['lifestyle', 'health', 'wellness', 'fashion', 'travel', 'fitness', 'home', 'food', 'beauty'],
    'music': ['music', 'album', 'concert', 'artist', 'band', 'music festival', 'songs', 'tour'],
    'finance': ['finance', 'stocks', 'investment', 'economy', 'market', 'business', 'cryptocurrency', 'real estate']
}

#Categorize News Articles
def categorize_article(article):
    title = article.get('title', '').lower()
    description = article.get('description', '').lower()
    content = article.get('content', '').lower()
    for category, keywords in category_keywords.items():
        if any(keyword in title for keyword in keywords) or \
           any(keyword in description for keyword in keywords) or \
           any(keyword in content for keyword in keywords):
            return category
    return 'miscellaneous'

#Fetching News Articles
newsapi = NewsApiClient(api_key=api_key)
API_parameters = {'q': 'australia OR sports OR lifestyle OR music OR finance', 'sources': sources_string, 'language': 'en'}
all_articles = newsapi.get_everything(**API_parameters)

if all_articles.get('status') == 'ok':
    for article in all_articles.get('articles', []):
        article_data = {
                'title': article['title'],
                'source': article['source']['name'],
                'publishedAt': article['publishedAt'],
                'url': article['url'],
                'description': article['description'],
                'content': article['content'],
                'category': categorize_article(article)
        }
        filtered_articles.append(article_data)
        
#Grouping articles with respect to categories
category_dict = {}
for item in filtered_articles:
    category = item['category']
    if category not in category_dict:
        category_dict[category] = []
    category_dict[category].append(item)
    
#Random sampling of articles (upto 10 articles per category)
sampled_articles = []
for category, category_items in category_dict.items():
    sampled_articles.extend(random.sample(category_items, min(5, len(category_items))))

#Detecting Duplicate / Similar Articles    
def detect_similar_articles(articles):
             
    #Embeddings for the content of articles
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    #for article in st.session_state.sampled_articles:
    for article in articles:
        content = article['content']
        embedding = embedding_model.encode(content)
        article['embedding'] = embedding
         
    #Cosine Similarity Metric
    embeddings = [article['embedding'] for article in articles]
    similarity_matrix = cosine_similarity(embeddings)
    
    #Detecting Similar Articles using Cosine Similarity Scores
    threshold_similarity = 0.5         #Similarity percentage
    similar_pairs = []
    for i in range(len(similarity_matrix)):
        for j in range(i + 1, len(similarity_matrix)):
            if similarity_matrix[i][j] >= threshold_similarity:
                similar_pairs.append((articles[i]['title'], articles[j]['title'], similarity_matrix[i][j]))
    
    st.session_state.similar_articles = similar_pairs
    return similar_pairs

#Detecting News Highlights
def detect_highlights():
    if not sampled_articles:
        st.warning("Please fetch news articles first!")
        return
    
    keywords = ["Breaking News", "Latest", "Urgent", "Update", "Exclusive", "Hot", "Alert", "Developing", "Now", "Must Read"]
    highlights = [
        article for article in sampled_articles
        if any(keyword.lower() in (article['title'] + article['description'] + article['content']).lower() for keyword in keywords)
    ]
    st.session_state.highlights = highlights
    return highlights

# Setting up the LLM
os.environ["GROQ_API_KEY"] = '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'
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1)

# Summarize article content
def summarize_article(content):
    messages = [{"role": "system", "content": "You are an AI assistant that summarizes articles in a few sentences."},
                {"role": "user", "content": f"You are an AI assistant that summarizes articles in a few sentences:\n\n{content}"}]
    
    response = llm.invoke(messages)
    summary = response.content
    return summary

#Main Title
st.title("Australian News Desk")

#Fetch News Article Button
if st.button("Fetch News Articles"):
    with st.spinner("Fetching latest news..."):
        if sampled_articles:
            st.success("News articles fetched successfully!")
            st.write("### Latest News Articles")
            for article in sampled_articles:
                with st.expander(f"{article['title']} ({article['category']})"):
                    st.write(f"**Source:** {article['source']}")
                    st.write(f"**Published At:** {article['publishedAt']}")
                    st.write(f"**Description:** {article['description']}")
                    st.write(f"[Read more]({article['url']})")
        else:
            st.error("No news articles found. Try again later.")

#Detect Duplicate / Similar Articles Button    
if st.button("Detect Similar Articles"):
    if sampled_articles:
        with st.spinner("Detecting similar articles using cosine similarity metric..."):
            similar_articles = detect_similar_articles(sampled_articles)
            if similar_articles:
                st.write("### Detected Similar Articles")
                for pair in similar_articles:
                    st.write(f"**Article 1:** {pair[0]}")
                    st.write(f"**Article 2:** {pair[1]}")
                    st.write(f"**Similarity Score:** {pair[2]:.4f}")
                    st.write("---")
            else:
                st.info("No similar article pairs found.")
    else:
        st.error("Please fetch news articles first.")

#Detect News Highlights         
if st.button("News Highlights"):
    if sampled_articles:
        with st.spinner("Scanning for breaking news, urgent news, important news...."):
            highlights = detect_highlights()
            if highlights:
                st.subheader("News Highlights")
                for idx, article in enumerate(st.session_state.highlights):
                    st.markdown(f"**{article['title']}** - {article['source']}")
                    st.markdown(f"[Read more]({article['url']})")
                    st.write("---")
    else:
        st.error("Please fetch news articles first.")
                                     
#Fetch News Summaries Button
if st.button("News Summaries"):
    if 'highlights' in st.session_state and st.session_state.highlights:
        with st.spinner("Generating Summaries using LLM..."):
            for highlight in st.session_state.highlights:
                summary = summarize_article(highlight['content'])
            
                # Display the summary
                st.markdown(f"**{highlight['title']}** - {highlight['source']}")
                st.write(f"**Summary**: {summary}")
                st.markdown(f"[Read full article]({highlight['url']})")
                st.write("---")
            
    else:
        st.warning("No highlights found. Please get highlights first.")
        

#Backend for RAG

#Preparation for Text Extraction 
def format_article(article):
    """
    Extract the title, description, and content from the article and format it with a unique ID.
    
    Parameters:
        article (dict): The article containing title, description, and content.
        
    Returns:
        str: A string containing the unique ID, title, description, and content.
    """
    
    # Extract title, description, and content
    title = article.get('title', 'No Title')
    description = article.get('description', 'No Description')
    content = article.get('content', 'No Content')
    
    # Format the output as a string
    formatted_article = f" \nTitle: {title}\nDescription: {description}\nContent:\n{content}\n"
    
    return formatted_article

def concatenate_all_articles(extracted_articles):
    all_articles = ""
    
    for article in extracted_articles:
        # Format each article and concatenate to the final string
        formatted_article = format_article(article)
        all_articles += formatted_article  # Append each formatted article to the final string
    
    return all_articles

#Chat Function
def get_response(user_query):
    response = retrieval_chain({
        "question": user_query,
        "chat_history": conversation_history
    })
    conversation_history.append((user_query, response['answer']))
    return response['answer']

#Setting up the Vector Database / Store
if sampled_articles:
    # Concatenate all news articles
    extracted_text = concatenate_all_articles(sampled_articles)
    #Text Chunking using Semantic Chunking 
    embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    text_splitter = SemanticChunker(embedding_model, breakpoint_threshold_type="percentile")
    semantic_chunks = text_splitter.create_documents([extracted_text])
    
    #Creating Embeddings and Setting up Vector Store
    embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    vector_store = Chroma(collection_name="articles",     embedding_function=embedding_model, persist_directory="./news_db")

    # Extract the content from the semantic chunks and convert to a list of strings
    semantic_chunk_texts = [chunk.page_content for chunk in semantic_chunks]
    vector_store.add_texts(texts=semantic_chunk_texts)
    
    #Building the retrieval chain
    retrieval_chain = ConversationalRetrievalChain.from_llm(llm=llm,
    retriever=vector_store.as_retriever(topk=3),
    return_source_documents=True)
    
    conversation_history = []        
        
# Define a function to handle the chat
def chat_with_llm():
    st.title("Chat with News")
    
    # Set up the conversation history (store in session state to persist across interactions)
    if 'conversation_history' not in st.session_state:
        st.session_state['conversation_history'] = []

    # Input box for the user to enter a query
    user_query = st.text_input("Ask a question:", "")
    
    if user_query:
        with st.spinner("Generating response..."):
            # Get the response from the semantic search and LLM
            response = get_response(user_query)
            
            # Display the conversation so far
            st.session_state['conversation_history'].append(('User', user_query))
            st.session_state['conversation_history'].append(('AI', response))
            
            # Show the conversation history
            for speaker, message in st.session_state['conversation_history']:
                st.markdown(f"**{speaker}:** {message}")
            st.write("---")

# Call the function to handle the chat UI
chat_with_llm()

#Backend for AI Agent

# Initialize the Groq client
client = Groq()
# Specify the model to be used (we recommend Llama 3.3 70B)
MODEL = 'llama-3.3-70b-versatile'

#Defining Web Search Tool
def search_web(query: str) -> str:
    """Search the web for news"""
    return DuckDuckGoSearchResults(backend="news").run(query)

#Defining Yahoo Finance Tool
def search_yf(query: str) -> str:
    """Search Yahoo Finance for financial content"""
    engine = DuckDuckGoSearchResults(backend="news")
    return engine.run(f"site:finance.yahoo.com {query}")

# imports search_web function 
def run_conversation(user_prompt):
    # Initialize the conversation with system and user messages
    messages=[
        {
            "role": "system",
            "content": "You are a search assistant. Use the search_web tool to search news, search_yf tool to search financial content, and provide the results. Must provide the URLs also."
        },
        {
            "role": "user",
            "content": user_prompt,
        }
    ]
    # Define the available tools (i.e. functions) for our model to use
    tools = [
        {
            "type": "function",
            "function": {
                "name": "search_web",
                "description": "Search the web for news",
                "parameters": {
                    "type": "object", 'required':['query'],
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The topic or subject to search on the web",
                        }
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_yf",
                "description": "Search yahoo finance for financial content",
                "parameters": {
                    "type": "object", 'required':['query'],
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The financial topic or subject to search",
                        }
                    },
                    "required": ["query"],
                },
            },
        },
        
        
    ]
    # Make the initial API call to Groq
    response = client.chat.completions.create(
        model=MODEL, # LLM to use
        messages=messages, # Conversation history
        stream=False,
        tools=tools, # Available tools (i.e. functions) for our LLM to use
        tool_choice="auto", # Let our LLM decide when to use tools
        max_completion_tokens=4096 # Maximum number of tokens to allow in our response
    )
    # Extract the response and any tool call responses
    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls
    if tool_calls:
        # Define the available tools that can be called by the LLM
        available_functions = {
            "search_web": search_web,
            "search_yf": search_yf,
            
        }
        # Add the LLM's response to the conversation
        messages.append(response_message)

        # Process each tool call
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_to_call = available_functions[function_name]
            function_args = json.loads(tool_call.function.arguments)
            # Call the tool and get the response
            function_response = function_to_call(
                query=function_args.get("query")
            )
            # Add the tool response to the conversation
            messages.append(
                {
                    "tool_call_id": tool_call.id, 
                    "role": "tool", # Indicates this message is from tool use
                    "name": function_name,
                    "content": function_response,
                }
            )
        # Make a second API call with the updated conversation
        second_response = client.chat.completions.create(
            model=MODEL,
            messages=messages
        )
        # Return the final response
        return second_response.choices[0].message.content

# Define a function to handle the agent
def agentic_news():
    st.title("AI Agent for News")
    
    # Create a text input box for the user to input a query
    user_query = st.text_area("Ask a question or search news:", "")
    
    if user_query:
        with st.spinner("Agentic Search..."):
            # Get response from the AI Agent using the provided backend function
            try:
                response = run_conversation(user_query)
                st.subheader("Search Results")
                st.write(response)
            except Exception as e:
                st.error("An error occured. Please re try")
    else:
        st.warning("Please enter a query.")
                  
#Call the function to activate the agent
agentic_news()

# Footer Bar
st.markdown("<h3 style='text-align: center; color: white;'>Thank You</h3>", unsafe_allow_html=True)


