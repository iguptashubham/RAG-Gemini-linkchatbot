import streamlit as st 
from main import LLMChain

llmchain = LLMChain()

page_bg_img = '''
<style>
[data-testid="stAppViewContainer"] > .main {
    background-image: url("https://unsplash.com/photos/orange-and-blue-color-illustration-p-NQlmGvFC8");
    background-size: 180%;
    background-position: top left;
    background-repeat: no-repeat;
    background-attachment: local;
}
[data-testid="stSidebar"] > div:first-child {
    background: none; /* Remove the second background image */
}
[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}
[data-testid="stToolbar"] {
    right: 2rem;
}
</style>
'''


st.set_page_config(page_title="Custom LLM Model", layout='wide')
st.markdown(page_bg_img,unsafe_allow_html=True)
st.title("Custom :blue[Gemini] Model")

c1,c2 = st.columns([0.3,0.7])

with c1:
    with st.container(border=True):
        st.write('Paste your URL here')
        link1 = st.text_input(label='link1')
        link2 = st.text_input(label='link2')
        link3 = st.text_input(label='link3')
        button = st.button('Start')
        link=[]
        if link1:
            link.append(link1)
        elif link2:
            link.append(link2)
        elif link3:
            link.append(link3)
        if button:
            data = llmchain.data_collection(link)
            split_data = llmchain.data_split(data)
            vectordb = llmchain.vectordb(split_data, add=True)
            

with c2:
    with st.container(border=True):
        user_query = st.chat_input('Enter your query here')
        if user_query:
            vectordb = llmchain.vectordb(add=False)
            prompt = llmchain.prompts()
            output = llmchain.rag(query=user_query, prompt=prompt,db=vectordb)
            with st.chat_message('human'):
                st.write(user_query)
            with st.chat_message('ai'):
                st.write(output)
            

