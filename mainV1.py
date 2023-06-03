# Imports
import os
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.llms import OpenAI
import streamlit as st
from htmlTemplates import css, bot_template, user_template
from apikeys import serpapikey, openaikey

def main():
    # Streamlit page configuration
    st.set_page_config(page_title='Serpapi Agent', layout='wide')

    # Apply CSS styling
    st.write(css, unsafe_allow_html=True)

    # Initialize Session States
    if 'generated' not in st.session_state: 
        st.session_state['generated'] = []

    if 'past' not in st.session_state: 
        st.session_state['past'] = []

    # Set up the Streamlit app layout
    st.title("Intelligent Google Search ChatBot")
    st.subheader(" Powered by LangChain + OpenAI + Serpapi + Streamlit") 


    # Define Language Model
    llm = OpenAI(model_name='gpt-3.5-turbo',
                temperature=0.9,
                max_tokens=256)
    
    # Load in some tools to use - serpapi for google use and llm-math for math ops
    tools = load_tools(["serpapi", "llm-math"], llm=llm)

    # Initialize ageny
    agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)

    # Accept input from user
    user_input = st.text_input("Enter your message:")  

    # Submit Button Logic
    if st.button("Submit") and user_input:
        with st.spinner('Generating response...'):
            try:
                # Generate response
                try:
                    response = agent.run(user_input)
                except:
                    response = "Sorry I am unable to answer your question"

                # Store conversation
                st.session_state.past.append(user_input)
                st.session_state.generated.append(response)

                # Display conversation
                for i in range(len(st.session_state.past)-1,-1,-1):
                    st.write(bot_template.replace("{{MSG}}",st.session_state.generated[i] ), unsafe_allow_html=True)
                    st.write(user_template.replace("{{MSG}}",st.session_state.past[i] ), unsafe_allow_html=True)
                    st.write("")

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")


if __name__ == '__main__':
    os.environ['OPENAI_API_KEY'] = openaikey
    os.environ["SERPAPI_API_KEY"] = serpapikey
    main()
