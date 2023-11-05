import os
import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper
from apikey import apikey

os.environ['OPENAI_API_KEY'] = apikey

st.title('Youtube GPT creator!')
prompt = st.text_input('Plug in your prompt here')
title_template = PromptTemplate(
    input_variables=['topic'],
    template='write me a youtube video title about {topic}'
)
script_template = PromptTemplate(
    input_variables=['title', 'wikipedia_research'],
    template='write me a youtube video script based in this TITLE:{title} while leveraging wikipedia research {wikipedia_research}'
)

#Memory
title_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
script_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')

#Llms
llm = OpenAI(temperature=0.9)
title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key='title', memory=title_memory)
script_chain = LLMChain(llm=llm, prompt=script_template, verbose=True, output_key='script', memory=script_memory)

wiki = WikipediaAPIWrapper()
#sequential_chain=SequentialChain (chains=[title_chain,script_chain], input_variables=['topic'],
#                                output_variables=['title', 'script'], verbose=True)

#shows result if there is a prompt

if prompt:
    title = title_chain.run(prompt)
    wiki_research = wiki.run(prompt)
    script = script_chain.run(title=title, wikipedia_research=wiki_research)

    #response = sequential_chain({'topic':prompt})
    st.write(title) 
    st.write(script)

    with st.expander('Title history'):
        st.info(title_memory.buffer)
    
    with st.expander('Script history'):
        st.info(script_memory.buffer)
