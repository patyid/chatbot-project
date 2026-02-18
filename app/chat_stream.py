"""
Chatbot com Streamlit + LangChain + Ollama (ChatOllama)
- UI: Streamlit (chat_input + chat_message)
- LLM local: Ollama via HTTP (ex.: http://localhost:11434)
- Memória: SQLChatMessageHistory (SQLite) por session_id (user_id)
- Histórico na tela: st.session_state.chat_history
"""

import streamlit as st
import os

from dotenv import load_dotenv  # Carrega variáveis de ambiente do arquivo .env (pode trocar por Langfuse/Opik depois)
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
# Interface (Streamlit)
# ---------------------------
#insira 
st.title("Meu primeiro Chatbot")


# Config do Ollama (LLM local)
base_url = "http://localhost:11434"
base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
model = os.getenv("OLLAMA_MODEL", "llama3.2")  # ← lê da variável de ambiente

st.caption(f"🤖 Modelo ativo: **{model}**")  # opcional: mostra qual modelo está ativo

# ---------------------------
# Captura do user_id (servirá como session_id para a memória no SQLite)
# ---------------------------
with st.form("user_form"):
    user_id = st.text_input("Digite seu ID de usuário", key="user_id")
    submit_button = st.form_submit_button("Enviar")

# Validação simples do user_id
if submit_button:
    if not user_id:
        st.error("Por favor, insira um ID de usuário para continuar. O histórico da conversa será salvo com base nesse ID.")
    else:
        st.write(f"User id: {user_id}")

# ---------------------------
# Memória: histórico persistente no SQLite
# ---------------------------
def get_session_history(session_id: str) -> SQLChatMessageHistory:
    """
    Retorna o histórico de mensagens para um session_id.
    Internamente usa SQLite via SQLChatMessageHistory.

    Args:
        session_id (str): id da sessão (aqui: user_id).

    Returns:
        SQLChatMessageHistory: objeto com histórico persistido.
    """
    return SQLChatMessageHistory(session_id, connection="sqlite:///chat_history.db")


# ---------------------------
# Histórico em memória (para renderizar na UI)
# ---------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Botão para resetar conversa:
# - limpa o histórico da UI (session_state)
# - apaga o histórico persistido no SQLite (history.clear())
if st.button("Iniciar nova conversa"):
    st.session_state.chat_history = []
    if user_id:  # evita quebrar se clicar antes de informar user_id
        history = get_session_history(user_id)
        history.clear()

# Renderiza o histórico da UI
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ---------------------------
# Setup do LLM + Prompt + Chain
# ---------------------------
llm = ChatOllama(base_url=base_url, model=model)

# Mensagem de sistema: define comportamento do assistente
system = SystemMessagePromptTemplate.from_template("Você é um assistente prestativo. Responda sempre em português (pt-BR).")

# Mensagem humana: onde entra o input do usuário
human = HumanMessagePromptTemplate.from_template("{input}")

# MessagesPlaceholder injeta o histórico no prompt (memória)
messages = [system, MessagesPlaceholder(variable_name="history"), human]

# Prompt final (system + history + human)
prompt_template = ChatPromptTemplate(messages=messages)

# Chain:
# prompt -> LLM -> parser (texto)
chain = prompt_template | llm | StrOutputParser()

# Wrapper que conecta a chain com histórico persistente
runnable_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",     # chave do dicionário de entrada
    history_messages_key="history", # nome usado no MessagesPlaceholder
)

def chat_with_llm(session_id: str, user_input: str):
    """
    Faz streaming da resposta do modelo usando histórico persistente.

    Observação:
    - stream() vai emitindo pedaços (tokens/chunks) de texto.
    - o config/configurable/session_id define qual histórico (SQLite) usar.
    """
    for output in runnable_with_history.stream(
        {"input": user_input},
        config={"configurable": {"session_id": session_id}},
    ):
        yield output


# ---------------------------
# Input do chat
# ---------------------------
user_prompt = st.chat_input("Ola, Como posso ajudar?")

if user_prompt:
    # Guarda mensagem do usuário na UI
    st.session_state.chat_history.append({"role": "user", "content": user_prompt})

    # Mostra mensagem do usuário
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Mostra resposta do assistente em streaming
    with st.chat_message("assistant"):
        # Se user_id estiver vazio, o histórico no SQLite não funcionará direito.
        # Aqui mantemos o comportamento simples: usa o que o usuário digitou.
        response = st.write_stream(chat_with_llm(user_id, user_prompt))

    # Guarda resposta do assistente na UI
    st.session_state.chat_history.append({"role": "assistant", "content": response})