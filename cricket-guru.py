import asyncio
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import streamlit as st
import os

st.set_page_config(
    page_title="Cricket Guru",
    page_icon="🏏",
)

openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")

os.environ["LANGCHAIN_API_KEY"] = st.secrets["langchain_api_key"]
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "cricket-bot"

st.title("Cricket Guru🏏")
st.caption("Your go-to assistant for cricket stats, trivia, and more!🚀")


if "openai_model" not in st.session_state:
    st.session_state.openai_model = "gpt-3.5-turbo"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


async def generate_response():
    model = ChatOpenAI(
        api_key=openai_api_key,
        model=st.session_state.openai_model,
        temperature=0.7,
        max_tokens=100,
        streaming=True,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content="You are a cricket expert. Strictly answer questions only related to cricket. Do not answer any other question. Also, for answers of cricket which are later than your training data, do not hallucinate. Just display the text describing that you don't know."
            ),
            MessagesPlaceholder(variable_name="question"),
        ]
    )

    output_parser = StrOutputParser()

    chain = prompt | model | output_parser

    messages = [
        (
            HumanMessage(content=msg["content"])
            if msg["role"] == "user"
            else AIMessage(content=msg["content"])
        )
        for msg in st.session_state.messages
    ]

    response_container = st.empty()
    response = ""

    async for partial_response in chain.astream({"question": messages}):
        response += partial_response
        response_container.write(response)

    return response


try:
    if prompt := st.chat_input("What's up?"):
        with st.chat_message("user"):
            st.write(prompt)

        if not openai_api_key.startswith("sk-"):
            st.warning("Please enter your OpenAI API key!", icon="⚠️")

        else:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("assistant"):
                response = asyncio.run(generate_response())

            st.session_state.messages.append({"role": "assistant", "content": response})

except Exception as e:
    with st.chat_message("error", avatar="⚠️"):
        st.write(f"Error: {e}")
