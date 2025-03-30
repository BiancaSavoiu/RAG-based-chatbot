from langchain_google_genai import ChatGoogleGenerativeAI
from FlagEmbedding import BGEM3FlagModel
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, RemoveMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
import os
import google.generativeai as genai
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
import streamlit as st

# Model Initialization
model = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0)

# Embedding Model
model_fp16 = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

class M3EmbeddingFP16:
    def embed_documents(self, texts):
        return model_fp16.encode(texts)['dense_vecs']
    
    def __call__(self, texts):
        return self.embed_documents(texts)
    
embd = M3EmbeddingFP16()

# Vector Store
vectorstore = FAISS.load_local("recursive_augmented_faiss_index", embd, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# Define Workflow
workflow = StateGraph(state_schema=MessagesState)

# Format documents for output
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Retrieve relevant documents
def retrieve_documents(question):
    docs = retriever.get_relevant_documents(question)
    return format_docs(docs)

# Call the model
def call_model(state: MessagesState):
    system_prompt = (
        """Comportati come un assistente che risponde alle domande del cliente.   
        Rispondi alla domanda basandoti solo sui seguenti documenti: {context}.
        Il contesto fornito può contenere anche il riassunto dei precedenti messaggi di questa conversazione.

        Rispondi in modo conciso e chiaro, spiegando passo passo al cliente le azioni necessarie da effettuare.   
        Se possibile, dai indicazioni dettagliate al cliente, su come risolvere il problema o effettuare l'azione desiderata. 
        Evita troppe ripetizioni nella risposta fornita.
        Quando spieghi che cosa è o cosa significa un certo elemento richiesto, non parlarne come se fosse un problema.

        In caso di più domande rispondi solo a quelle inerenti alla documentazione e rimani a disposizione per altre domande sull'argomento,
        specificando, invece, che le altre domande non sono state trovate pertinenti in questo contesto.

        Domanda relativa al software Panthera: {question} 
        """
    )

    message_history = state["messages"][:-1]  # Exclude most recent user input
    last_human_message = state["messages"][-1]
    context = retrieve_documents(last_human_message.content)

    prompt = system_prompt.format(context=context, question=last_human_message.content)
    system_message = SystemMessage(content=prompt)

    if len(message_history) >= 4:
        summary_prompt = (
            "Distilla i messaggi della chat sopra in un unico messaggio di riepilogo."
            "Includi il maggior numero possibile di dettagli specifici."
        )
        summary_message = model.invoke(
            message_history + [HumanMessage(content=summary_prompt)]
        )
        delete_messages = [RemoveMessage(id=m.id) for m in state["messages"]]
        human_message = HumanMessage(content=last_human_message.content)
        response = model.invoke([system_message, summary_message, human_message])
        message_updates = [summary_message, human_message, response] + delete_messages
    else:
        response = model.invoke([system_message] + state["messages"])
        message_updates = [response]

    return {"messages": message_updates}

# Add Workflow Node
workflow.add_node("model", call_model)
workflow.add_edge(START, "model")

# Add Memory Saver
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# Streamlit UI
st.title("💬 Chatbot con Panthera e RAG")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Ciao! Come posso aiutarti oggi?"}]

# Display chat history
for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

# User input
if user_input := st.chat_input(placeholder="Scrivi la tua domanda qui..."):
    st.session_state["messages"].append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    # Process input through the workflow
    state = {"messages": st.session_state["messages"]}
    try:
        response = app.invoke({"messages": state["messages"][-1]}, config={"configurable": {"thread_id": "1"}})
        # Extract content from the last message
        assistant_message = response["messages"][-1].content  # Adjusted for list of dictionaries
    except Exception as e:
        assistant_message = f"Errore: {str(e)}"

    # Display assistant's response and update chat history
    st.session_state["messages"].append({"role": "assistant", "content": assistant_message})
    st.chat_message("assistant").write(assistant_message)