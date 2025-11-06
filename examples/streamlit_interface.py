"""
Template para criar interface web com Streamlit
Execute: streamlit run streamlit_interface.py
"""
import streamlit as st
import sys
from pathlib import Path

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.agents import CandangoAgent
from src.utils import setup_logging, create_sample_documents
from src.config import config
import time
import uuid


# Configuração da página
st.set_page_config(
    page_title="🤖 Candango Lanches - IA Assistant",
    page_icon="🍔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado (alto contraste)
st.markdown("""
<style>
    /* Cabeçalho com contraste forte */
    .main-header {
        background: linear-gradient(90deg, #0D47A1 0%, #1976D2 100%);
        padding: 1.2rem;
        border-radius: 10px;
        color: #FFFFFF;
        text-align: center;
        margin-bottom: 1.5rem;
        box-shadow: 0 3px 6px rgba(0,0,0,0.18);
    }

    /* Container geral das mensagens */
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.6rem 0;
        box-shadow: 0 2px 6px rgba(0,0,0,0.12);
        font-size: 1rem;
        color: #111827; /* texto escuro para legibilidade */
    }

    /* Mensagens do usuário: fundo branco, borda azul escura */
    .user-message {
        background-color: #FFFFFF;
        border-left: 5px solid #0D47A1;
        color: #0B1220;
    }

    /* Mensagens do assistente: leve cinza de fundo com borda roxa escura */
    .assistant-message {
        background-color: #F5F7FA;
        border-left: 5px solid #4A148C;
        color: #0B1220;
    }

    /* Informações do sistema com destaque acessível */
    .system-info {
        background-color: #FFF8E1;
        border: 1px solid #FBC02D;
        border-radius: 6px;
        padding: 0.65rem;
        font-size: 0.9rem;
        color: #3E3A00;
    }

    /* Cartões de métricas com fundo branco e texto escuro */
    .metric-card {
        background: #FFFFFF;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.12);
        text-align: center;
        color: #111827;
    }

    /* Ajustes responsivos e de foco para acessibilidade */
    .stButton>button:focus {
        outline: 3px solid rgba(13,71,161,0.25);
        outline-offset: 2px;
    }

</style>
""", unsafe_allow_html=True)


@st.cache_resource
def inicializar_agente():
    """Inicializa o agente (cached para performance)"""
    try:
        # Setup logging silencioso para interface
        setup_logging("WARNING")
        
        # Verificar se há documentos
        if not list(config.documents_dir.glob("*.pdf")):
            with st.spinner("📄 Criando base de conhecimento..."):
                create_sample_documents(config.documents_dir)
        
        # Inicializar agente
        with st.spinner("🤖 Inicializando IA..."):
            agent = CandangoAgent()
        
        return agent, None
    except Exception as e:
        return None, str(e)


def mostrar_header():
    """Exibe cabeçalho da aplicação"""
    st.markdown("""
    <div class="main-header">
        <h1>🍔 Candango Lanches - Assistente IA</h1>
        <p>Sistema inteligente com RAG + LangGraph para atendimento automatizado</p>
    </div>
    """, unsafe_allow_html=True)


def mostrar_sidebar():
    """Configura sidebar com informações e controles"""
    with st.sidebar:
        st.image("https://via.placeholder.com/200x100/FF6B35/FFFFFF?text=Candango+Lanches", 
                caption="Lanchonete Candango Lanches")
        
        st.markdown("---")
        
        st.subheader("⚙️ Configurações")
        
        # Configurações do chat
        temperatura = st.slider(
            "Criatividade da IA", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.2, 
            step=0.1,
            help="0 = mais preciso, 1 = mais criativo"
        )
        
        mostrar_debug = st.checkbox(
            "Mostrar informações técnicas",
            value=False,
            help="Exibe detalhes da classificação e fontes"
        )
        
        # Botão para limpar conversa
        if st.button("🗑️ Limpar Conversa", use_container_width=True):
            st.session_state.messages = []
            st.session_state.session_id = str(uuid.uuid4())
            st.rerun()
        
        st.markdown("---")
        
        # Informações do sistema
        st.subheader("📊 Status do Sistema")
        
        # Verificar se agente está inicializado
        agent_status = "✅ Online" if st.session_state.get('agent') else "❌ Offline"
        st.metric("Agente IA", agent_status)
        
        # Número de mensagens na conversa
        num_messages = len(st.session_state.get('messages', []))
        st.metric("Mensagens", num_messages)
        
        st.markdown("---")
        
        # Exemplos de perguntas
        st.subheader("💡 Perguntas Exemplo")
        exemplos = [
            "Quais são os horários de funcionamento?",
            "Como devo limpar as fritadeiras?",
            "Tive um problema com meu pedido",
            "Posso fazer um lanche sem cebola?",
            "Qual é a política de higiene?"
        ]
        
        for exemplo in exemplos:
            if st.button(f"💬 {exemplo}", use_container_width=True, key=f"exemplo_{hash(exemplo)}"):
                st.session_state.messages.append({
                    "role": "user", 
                    "content": exemplo,
                    "timestamp": time.time()
                })
                st.rerun()
        
        return mostrar_debug


def mostrar_mensagem(message, mostrar_debug=False):
    """Exibe uma mensagem do chat com formatação"""
    role = message["role"]
    content = message["content"]
    timestamp = message.get("timestamp", time.time())
    
    # Formatação baseada no papel
    if role == "user":
        st.markdown(f"""
        <div class="chat-message user-message">
            <strong>👤 Você</strong><br>
            {content}
        </div>
        """, unsafe_allow_html=True)
    
    elif role == "assistant":
        st.markdown(f"""
        <div class="chat-message assistant-message">
            <strong>🤖 Candango Assistant</strong><br>
            {content}
        </div>
        """, unsafe_allow_html=True)
        
        # Mostrar informações de debug se habilitado
        if mostrar_debug and "debug_info" in message:
            debug_info = message["debug_info"]
            
            with st.expander("🔍 Informações Técnicas"):
                col1, col2 = st.columns(2)
                
                with col1:
                    if "triagem" in debug_info:
                        triagem = debug_info["triagem"]
                        st.write(f"**Classificação:** {triagem.get('decisao', 'N/A')}")
                        st.write(f"**Urgência:** {triagem.get('urgencia', 'N/A')}")
                
                with col2:
                    if "rag" in debug_info:
                        rag = debug_info["rag"]
                        st.write(f"**Contexto encontrado:** {'Sim' if rag.get('context_found') else 'Não'}")
                        if rag.get('sources'):
                            st.write(f"**Fontes:** {', '.join(rag['sources'])}")


def processar_mensagem(agent, user_input, session_id):
    """Processa mensagem do usuário com o agente"""
    try:
        with st.spinner("🤖 Processando..."):
            # Processar com agente
            state = agent.process_message(user_input, thread_id=session_id)
            
            # Extrair resposta do assistente
            assistant_messages = [msg for msg in state.messages if msg.role == "assistant"]
            
            if assistant_messages:
                response_content = assistant_messages[-1].content
                
                # Preparar informações de debug
                debug_info = {}
                
                if state.triagem_result:
                    debug_info["triagem"] = {
                        "decisao": state.triagem_result.decisao.value,
                        "urgencia": state.triagem_result.urgencia.value
                    }
                
                if state.rag_response:
                    debug_info["rag"] = {
                        "context_found": state.rag_response.context_found,
                        "sources": state.rag_response.sources
                    }
                
                return response_content, debug_info
            else:
                return "Desculpe, não consegui processar sua mensagem.", {}
                
    except Exception as e:
        st.error(f"Erro ao processar mensagem: {e}")
        return "Ocorreu um erro interno. Tente novamente.", {}


def main():
    """Função principal da aplicação Streamlit"""
    # Mostrar header
    mostrar_header()
    
    # Configurar sidebar
    mostrar_debug = mostrar_sidebar()
    
    # Inicializar estado da sessão
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    
    if "agent" not in st.session_state:
        agent, error = inicializar_agente()
        if agent:
            st.session_state.agent = agent
            st.success("✅ Sistema inicializado com sucesso!")
        else:
            st.error(f"❌ Erro ao inicializar sistema: {error}")
            st.stop()
    
    # Área principal do chat
    st.subheader("💬 Conversa")
    
    # Container para mensagens
    chat_container = st.container()
    
    # Exibir histórico de mensagens
    with chat_container:
        for message in st.session_state.messages:
            mostrar_mensagem(message, mostrar_debug)
    
    # Input do usuário usando formulário (evita submissões a cada alteração)
    with st.container():
        with st.form("chat_form", clear_on_submit=True):
            col1, col2 = st.columns([4, 1])

            with col1:
                user_input = st.text_input(
                    "Digite sua mensagem:",
                    placeholder="Ex: Quais são os horários de funcionamento?",
                    key="user_input_field"
                )

            with col2:
                submitted = st.form_submit_button("📤 Enviar", use_container_width=True)

        # Processar apenas quando o formulário for submetido
        if submitted and user_input and user_input.strip():
            # Adicionar mensagem do usuário
            st.session_state.messages.append({
                "role": "user",
                "content": user_input,
                "timestamp": time.time()
            })

            # Processar com agente
            response, debug_info = processar_mensagem(
                st.session_state.agent,
                user_input,
                st.session_state.session_id
            )

            # Adicionar resposta do assistente
            assistant_message = {
                "role": "assistant",
                "content": response,
                "timestamp": time.time()
            }

            if debug_info:
                assistant_message["debug_info"] = debug_info

            st.session_state.messages.append(assistant_message)

            # O form já limpa o campo quando clear_on_submit=True e a submissão causa rerun
    
    # Rodapé
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        🤖 Powered by RAG + LangGraph | 
        🔧 Built with Streamlit | 
        ⚡ Google Gemini API
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()