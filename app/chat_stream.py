"""
Chatbot com Streamlit + LangChain + Ollama (ChatOllama)
- UI: Streamlit (chat_input + chat_message)
- LLM local: Ollama via HTTP (ex.: http://localhost:11434)
- Memória: SQLChatMessageHistory (SQLite) por session_id (user_id)
- Histórico na tela: st.session_state.chat_history
"""

import os
import streamlit as st

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.output_parsers import StrOutputParser


# ---------------------------
# Config geral
# ---------------------------
load_dotenv()

st.title("Meu primeiro Chatbot")

base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
model = os.getenv("OLLAMA_MODEL", "llama3.2")
st.caption(f"Modelo ativo: **{model}**")


# ---------------------------
# Estado inicial
# ---------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "user_id_ok" not in st.session_state:
    st.session_state.user_id_ok = False

if "user_id" not in st.session_state:
    st.session_state.user_id = ""


# ---------------------------
# Função auxiliar: carregar histórico do SQLite
# ---------------------------
def load_chat_history_from_db():
    """Carrega mensagens do SQLite para o session_state"""
    if not st.session_state.get("user_id") or not st.session_state.get("user_id_ok"):
        return
    
    history = get_session_history(st.session_state.user_id)
    st.session_state.chat_history = []
    
    for msg in history.messages:
        role = "user" if msg.type == "human" else "assistant"
        st.session_state.chat_history.append({
            "role": role,
            "content": msg.content
        })


# ---------------------------
# Memória: histórico persistente no SQLite
# ---------------------------
def get_session_history(session_id: str) -> SQLChatMessageHistory:
    db_path = os.getenv("CHAT_HISTORY_DB", "chat_history.db")
    connection_string = f"sqlite:///{db_path}?check_same_thread=False"
    return SQLChatMessageHistory(session_id, connection=connection_string)


# ---------------------------
# Captura e validação do user_id (session_id)
# ---------------------------
with st.form("user_form"):
    user_id_input = st.text_input(
        "Digite seu ID de usuário",
        value=st.session_state.user_id,
        max_chars=10,
        help="O histórico da conversa será salvo com base nesse ID. Use um ID único para cada usuário.",
    )
    submit_button = st.form_submit_button("Enviar", use_container_width=True)

if submit_button:
    user_id_clean = (user_id_input or "").strip()

    if len(user_id_clean) < 3:
        st.session_state.user_id_ok = False
        st.error("O ID precisa ter pelo menos 3 caracteres.")
    elif len(user_id_clean) > 10:
        st.session_state.user_id_ok = False
        st.error("O ID precisa ter no máximo 10 caracteres.")
    else:
        st.session_state.user_id = user_id_clean
        st.session_state.user_id_ok = True
        st.success(f"User id: {user_id_clean}")
        # Carrega histórico existente do banco
        load_chat_history_from_db()
        st.rerun()


# ---------------------------
# Botão: iniciar nova conversa
# ---------------------------
if st.button("Iniciar nova conversa", use_container_width=True):
    st.session_state.chat_history = []
    if st.session_state.user_id_ok and st.session_state.user_id:
        history = get_session_history(st.session_state.user_id)
        history.clear()
    st.rerun()


# ---------------------------
# Renderiza histórico da UI
# ---------------------------
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ---------------------------
# Setup do LLM + Prompt + Chain
# ---------------------------
llm = ChatOllama(base_url=base_url, model=model)

system = SystemMessagePromptTemplate.from_template(
    "Você é um assistente prestativo. Responda sempre em português (pt-BR)."
)
human = HumanMessagePromptTemplate.from_template("{input}")

prompt_template = ChatPromptTemplate(messages=[
    system,
    MessagesPlaceholder(variable_name="history"),
    human,
])

chain = prompt_template | llm | StrOutputParser()

runnable_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

def chat_with_llm(session_id: str, user_input: str):
    for output in runnable_with_history.stream(
        {"input": user_input},
        config={"configurable": {"session_id": session_id}},
    ):
        yield output


# ---------------------------
# Input do chat (bloqueado até user_id válido)
# ---------------------------
if not st.session_state.user_id_ok:
    st.info("Informe um ID de usuário válido (3 a 10 caracteres) para iniciar o chat.")
    st.stop()

user_prompt = st.chat_input("Olá, como posso ajudar?")

if user_prompt:
    st.session_state.chat_history.append({"role": "user", "content": user_prompt})

    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        response = st.write_stream(
            chat_with_llm(st.session_state.user_id, user_prompt)
        )

    st.session_state.chat_history.append({"role": "assistant", "content": response})