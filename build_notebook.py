import json

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Olá, Dev! Bem-vindo(a) ao seu Workshop de RAG!\\n",
                "\\n",
                "**Seu papel:** Engenheiro(a) de IA Júnior\\n",
                "**Sua missão:** Construir e entender um chatbot para a **Lanchonete Candango Lanches** usando o poder de LLMs com uma base de conhecimento própria.\\n",
                "\\n",
                "Neste notebook, vamos explorar passo a passo como construir um chatbot que não apenas conversa, mas que também consulta documentos para dar respostas precisas. Este processo é chamado de **RAG (Retrieval-Augmented Generation)**, ou Geração Aumentada por Recuperação.\\n",
                "\\n",
                "**O que vamos aprender:**\\n",
                "1.  **Configurar o ambiente** com as bibliotecas essenciais como LangChain e Google Generative AI.\\n",
                "2.  **Construir um pipeline de RAG completo:** Desde carregar documentos até a geração de respostas baseadas neles.\\n",
                "3.  **Interagir com o chatbot** de forma prática.\\n",
                "4.  **Explorar tópicos avançados (Bônus):** Veremos como usar LLMs para tarefas de classificação (triagem) e daremos dicas valiosas para levar seu projeto para o próximo nível (produção).\\n",
                "\\n",
                "Vamos começar essa jornada!"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Parte 1: Configuração do Ambiente\\n",
                "\\n",
                "Nesta primeira parte, vamos preparar nosso ambiente de desenvolvimento, instalando as bibliotecas necessárias e configurando o acesso ao modelo de linguagem do Google, o Gemini."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Instalando todas as dependências necessárias de uma só vez.\\n",
                "# Usamos o '-q' para uma instalação mais limpa (quiet).\\n",
                "!pip install -q langchain langchain-google-genai google-generativeai langchain_community faiss-cpu langchain-text-splitters pymupdf reportlab"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Organizando todas as importações em um único lugar para maior clareza.\\n",
                "import os\\n",
                "from pathlib import Path\\n",
                "from typing import Literal, List, Dict, Any\\n",
                "\\n",
                "# LangChain e Google Generative AI\\n",
                "from google.colab import userdata\\n",
                "from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings\\n",
                "from langchain.chains.combine_documents import create_stuff_documents_chain\\n",
                "from langchain_core.prompts import ChatPromptTemplate\\n",
                "from langchain_core.runnables import RunnablePassthrough, RunnableParallel\\n",
                "from langchain_core.output_parsers import StrOutputParser\\n",
                "from langchain_core.messages import SystemMessage, HumanMessage\\n",
                "\\n",
                "# Carregamento e Divisão de Documentos\\n",
                "from langchain_community.document_loaders import PyMuPDFLoader\\n",
                "from langchain_text_splitters import RecursiveCharacterTextSplitter\\n",
                "\\n",
                "# Vector Store e Retriever\\n",
                "from langchain_community.vectorstores import FAISS\\n",
                "\\n",
                "# Pydantic para Modelagem de Dados (usado na seção bônus)\\n",
                "from pydantic import BaseModel, Field\\n",
                "\\n",
                "# ReportLab para criar PDFs (usado para gerar arquivos de exemplo)\\n",
                "from reportlab.pdfgen import canvas\\n",
                "from reportlab.lib.pagesizes import letter"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 1.1 - Configurando a Chave de API do Gemini\\n",
                "\\n",
                "Para usar os modelos do Google, você precisa de uma chave de API. A forma mais segura de fazer isso no Google Colab é usando o Gerenciador de Segredos.\\n",
                "\\n",
                "1.  Clique no ícone de chave 🔑 no painel esquerdo.\\n",
                "2.  Adicione um novo segredo com o nome `GEMINI_API_KEY`.\\n",
                "3.  Cole a sua chave de API do Google AI Studio no campo de valor.\\n",
                "4.  Certifique-se de que o acesso ao notebook está habilitado.\\n",
                "\\n",
                "O código abaixo acessará essa chave de forma segura."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "GOOGLE_API_KEY = userdata.get('GEMINI_API_KEY')\\n",
                "\\n",
                "# Inicializando o modelo de linguagem (LLM) que usaremos em todo o notebook.\\n",
                "# O 'temperature' controla a criatividade do modelo (0 = mais determinístico, 1 = mais criativo).\\n",
                "llm = ChatGoogleGenerativeAI(\\n",
                "    model=\\\"gemini-1.5-flash\\\",\\n",
                "    temperature=0.2,\\n",
                "    google_api_key=GOOGLE_API_KEY\\n",
                ")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Parte 2: Construindo o Pipeline de RAG\\n",
                "\\n",
                "Agora vamos ao coração do nosso projeto: o pipeline de RAG. Ele permitirá que nosso chatbot consulte documentos para formular suas respostas."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 2.1 - Preparando a Base de Conhecimento (Nossos Documentos)\\n",
                "\\n",
                "Um sistema RAG precisa de uma fonte de conhecimento. No nosso caso, serão as políticas da lanchonete em formato PDF.\\n",
                "\\n",
                "Para garantir que este notebook funcione para qualquer pessoa, vamos criar programaticamente alguns arquivos PDF de exemplo. Em um projeto real, você usaria seus próprios arquivos."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "print(\\\"Criando arquivos PDF de exemplo para a base de conhecimento...\\\")\\n",
                "\\n",
                "conteudo_atendimento = \\\"\\\"\\\"Política de Atendimento ao Cliente da Candango Lanches\\n",
                "- Boas-Vindas: A equipe deve sempre cumprimentar o cliente de forma amigável.\\n",
                "- Cardápio: O colaborador deve conhecer o cardápio para descrever os itens.\\n",
                "- Pedidos: Todos os pedidos devem ser conferidos com o cliente.\\n",
                "- Feedback: Reclamações e sugestões devem ser ouvidas com atenção e encaminhadas ao gerente.\\n",
                "\\\"\\\"\\\"\\n",
                "\\n",
                "conteudo_seguranca = \\\"\\\"\\\"Política de Segurança e Uso de Equipamentos da Candango Lanches\\n",
                "- Equipamentos de Proteção: O uso de calçados antiderrapantes é obrigatório.\\n",
                "- Manuseio de Facas: As facas devem ser guardadas em suportes específicos.\\n",
                "- Máquinas: O uso de cortadores e moedores só é permitido após treinamento.\\n",
                "- Limpeza de Fritadeiras: É proibido limpar fritadeiras com óleo quente.\\n",
                "\\\"\\\"\\\"\\n",
                "\\n",
                "conteudo_higiene = \\\"\\\"\\\"Política de Higiene e Segurança Alimentar da Candango Lanches\\n",
                "- Limpeza: Equipamentos e bancadas devem ser higienizados diariamente.\\n",
                "- Armazenamento: Alimentos perecíveis devem ser armazenados em temperaturas adequadas.\\n",
                "- Uniforme: É obrigatório o uso de uniforme limpo, touca e luvas.\\n",
                "- Contaminação Cruzada: Evitar o contato entre alimentos crus e cozidos.\\n",
                "\\\"\\\"\\\"\\n",
                "\\n",
                "arquivos_pdf = {\\n",
                "    \\\"Politica_Atendimento.pdf\\\": conteudo_atendimento,\\n",
                "    \\\"Politica_Seguranca.pdf\\\": conteudo_seguranca,\\n",
                "    \\\"Politica_Higiene.pdf\\\": conteudo_higiene\\n",
                "}\\n",
                "\\n",
                "for nome_arquivo, conteudo in arquivos_pdf.items():\\n",
                "    c = canvas.Canvas(nome_arquivo, pagesize=letter)\\n",
                "    textobject = c.beginText(50, 750)\\n",
                "    textobject.setFont(\\\"Times-Roman\\\", 12)\\n",
                "    for line in conteudo.splitlines():\\n",
                "        textobject.textLine(line)\\n",
                "    c.drawText(textobject)\\n",
                "    c.save()\\n",
                "\\n",
                "print(f\\\"{len(arquivos_pdf)} arquivos PDF de exemplo criados com sucesso!\\\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 2.2 - Carregando e Dividindo os Documentos\\n",
                "\\n",
                "Com os arquivos prontos, o primeiro passo do pipeline é carregá-los e dividi-los em pedaços menores (chunks). Por que dividir?\\n",
                "- **Contexto Limitado:** Modelos de linguagem têm um limite de quantos tokens conseguem processar de uma vez.\\n",
                "- **Busca Eficiente:** É mais fácil e rápido encontrar um trecho específico de informação em pedaços menores do que em um documento inteiro."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "docs = []\\n",
                "for pdf_path in Path(\\\"./\\\").glob(\\\"*.pdf\\\"):\\n",
                "    try:\\n",
                "        loader = PyMuPDFLoader(str(pdf_path))\\n",
                "        docs.extend(loader.load())\\n",
                "        print(f\\\"Documento carregado: {pdf_path}\\\")\\n",
                "    except Exception as e:\\n",
                "        print(f\\\"Erro ao carregar {pdf_path}: {e}\\\")\\n",
                "\\n",
                "# Dividindo os documentos em chunks.\\n",
                "# chunk_size: o tamanho máximo de cada pedaço.\\n",
                "# chunk_overlap: uma pequena sobreposição de texto entre os chunks para não perder o contexto.\\n",
                "splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=40)\\n",
                "chunks = splitter.split_documents(docs)\\n",
                "\\n",
                "print(f\\\"\\nTotal de documentos: {len(docs)}\\\")\\n",
                "print(f\\\"Total de chunks gerados: {len(chunks)}\\\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 2.3 - Criando a Memória Vetorial (Embeddings & Vector Store)\\n",
                "\\n",
                "Agora, precisamos converter nossos `chunks` de texto em vetores numéricos, ou **embeddings**. Embeddings capturam o significado semântico do texto, permitindo que comparemos o quão \\\"próximas\\\" são duas frases em termos de significado.\\n",
                "\\n",
                "Esses vetores serão armazenados em um **Vector Store**, um tipo de banco de dados otimizado para buscas de similaridade. Usaremos o **FAISS**, uma biblioteca do Facebook AI para buscas eficientes."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Inicializa o modelo de embeddings do Google.\\n",
                "embeddings = GoogleGenerativeAIEmbeddings(\\n",
                "    model=\\\"models/embedding-001\\\",\\n",
                "    google_api_key=GOOGLE_API_KEY\\n",
                ")\\n",
                "\\n",
                "# Cria o banco de dados vetorial FAISS a partir dos chunks e embeddings.\\n",
                "# Adicionamos uma verificação para garantir que temos chunks antes de criar o índice.\\n",
                "if chunks:\\n",
                "    vectorstore = FAISS.from_documents(chunks, embeddings)\\n",
                "    print(\\\"Banco de dados vetorial FAISS criado com sucesso!\\\")\\n",
                "else:\\n",
                "    print(\\\"Nenhum chunk foi gerado. O banco de dados vetorial não pode ser criado.\\\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 2.4 - O Recuperador (The Retriever)\\n",
                "\\n",
                "O Retriever é o componente que, dada uma pergunta do usuário, faz a busca no Vector Store e \\\"recupera\\\" os chunks mais relevantes para responder àquela pergunta."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Converte o vectorstore em um retriever.\\n",
                "# search_type=\\\"similarity_score_threshold\\\": busca por similaridade e filtra por um score mínimo.\\n",
                "# search_kwargs: define o score mínimo (0.3) e o número máximo de documentos a retornar (k=4).\\n",
                "if 'vectorstore' in locals() and vectorstore:\\n",
                "  retriever = vectorstore.as_retriever(\\n",
                "      search_type=\\\"similarity_score_threshold\\\",\\n",
                "      search_kwargs={\\\"score_threshold\\\": 0.3, \\\"k\\\": 4}\\n",
                "  )\\n",
                "  print(\\\"Retriever configurado.\\\")\\n",
                "else:\\n",
                "  retriever = None\\n",
                "  print(\\\"Retriever não pôde ser configurado pois o vectorstore não foi criado.\\\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 2.5 - Criando a Cadeia de RAG com LCEL\\n",
                "\\n",
                "Agora, vamos juntar tudo! Usaremos a **LangChain Expression Language (LCEL)** para criar uma \\\"cadeia\\\" de operações:\\n",
                "\\n",
                "1.  A pergunta do usuário é passada para o `retriever`.\\n",
                "2.  O `retriever` busca os `chunks` relevantes (o **contexto**).\\n",
                "3.  A pergunta original e o contexto recuperado são passados para um **prompt**.\\n",
                "4.  O prompt preenchido é enviado para o **LLM**.\\n",
                "5.  O LLM gera a resposta final."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Template do prompt que instrui o LLM sobre como agir.\\n",
                "prompt_rag = ChatPromptTemplate.from_messages([\\n",
                "    (\\\"system\\\",\\n",
                "     \\\"Você é o assistente virtual da Lanchonete Candango Lanches. \\\"\\n",
                "     \\\"Sua função é responder a perguntas sobre as políticas da lanchonete com base no contexto fornecido. \\\"\\n",
                "     \\\"Seja amigável, prestativo e use uma linguagem casual e acolhedora. \\\"\\n",
                "     \\\"Responda APENAS com base nas informações do contexto abaixo.\\\\n\\\\n\\\"\\n",
                "     \\\"Contexto:\\\\n{context}\\\"\\n",
                "     ),\\n",
                "    (\\\"human\\\", \\\"Pergunta: {question}\\\"),\\n",
                "])\\n",
                "\\n",
                "# Cria a cadeia que combina o LLM e o prompt.\\n",
                "chain_de_geracao = create_stuff_documents_chain(llm, prompt_rag)\\n",
                "\\n",
                "# Cria a cadeia de RAG completa usando LCEL.\\n",
                "# RunnableParallel permite que o retriever e a pergunta sejam processados em paralelo.\\n",
                "# .assign(answer=...) adiciona a resposta gerada ao resultado final.\\n",
                "if retriever:\\n",
                "  rag_chain = RunnableParallel(\\n",
                "      {\\\"context\\\": retriever, \\\"question\\\": RunnablePassthrough()}\\n",
                "  ).assign(answer = chain_de_geracao | StrOutputParser())\\n",
                "  print(\\\"Cadeia de RAG criada com sucesso!\\\")\\n",
                "else:\\n",
                "  rag_chain = None\\n",
                "  print(\\\"Cadeia de RAG não foi criada devido a erro no retriever.\\\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 2.6 - A Função de Consulta Final\\n",
                "\\n",
                "Para facilitar a interação, vamos encapsular a chamada da `rag_chain` em uma função que também formata a saída, incluindo as fontes consultadas."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "def perguntar_ao_rag(pergunta: str) -> Dict[str, Any]:\\n",
                "    \\\"\\\"\\\"\\n",
                "    Recebe uma pergunta, executa o pipeline de RAG e retorna a resposta com citações.\\n",
                "    \\\"\\\"\\\"\\n",
                "    if not rag_chain:\\n",
                "        return {\\n",
                "            \\\"answer\\\": \\\"Desculpe, o sistema de busca não está operacional.\\\",\\n",
                "            \\\"citacoes\\\": [],\\n",
                "            \\\"contexto_encontrado\\\": False\\n",
                "        }\\n",
                "        \\n",
                "    resultado = rag_chain.invoke(pergunta)\\n",
                "    \\n",
                "    # Usamos .get() para acessar as chaves do dicionário de forma segura.\\n",
                "    answer = resultado.get('answer', 'Não foi possível obter uma resposta.')\\n",
                "    docs_relacionados = resultado.get('context', [])\\n",
                "\\n",
                "    # Formata as citações dos documentos recuperados.\\n",
                "    citacoes = [Path(doc.metadata.get('source', '')).name for doc in docs_relacionados]\\n",
                "    \\n",
                "    # Lógica de controle: se nenhum documento foi encontrado, a resposta é genérica.\\n",
                "    if not docs_relacionados:\\n",
                "        return {\\n",
                "            \\\"answer\\\": \\\"Desculpe, não encontrei informações sobre isso em nossas políticas.\\\",\\n",
                "            \\\"citacoes\\\": [],\\n",
                "            \\\"contexto_encontrado\\\": False\\n",
                "        }\\n",
                "    \\n",
                "    return {\\n",
                "        \\\"answer\\\": answer,\\n",
                "        \\\"citacoes\\\": list(set(citacoes)), # Remove duplicatas\\n",
                "        \\\"contexto_encontrado\\\": True\\n",
                "    }"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Parte 3: Interagindo com o Chatbot\\n",
                "\\n",
                "Tudo pronto! Agora vamos criar um loop interativo para conversar com nosso chatbot."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "def chatbot_lanchonete():\\n",
                "    print(\\\"\\\\n--- Bem-vindo ao Chatbot da Candango Lanches! ---\\\")\\n",
                "    print(\\\"Posso responder perguntas sobre nossas políticas de atendimento, segurança e higiene.\\\")\\n",
                "    print(\\\"Digite 'sair' para encerrar.\\\")\\n",
                "    print(\\\"-\\\" * 50)\\n",
                "\\n",
                "    while True:\\n",
                "        pergunta_cliente = input(\\\"Sua pergunta: \\\")\\n",
                "        if pergunta_cliente.lower() == 'sair':\\n",
                "            print(\\\"\\\\nObrigado por usar o Chatbot da Candango Lanches!\\\")\\n",
                "            break\\n",
                "\\n",
                "        resultado = perguntar_ao_rag(pergunta_cliente)\\n",
                "\\n",
                "        print(\\\"\\\\nResposta do Chatbot:\\\")\\n",
                "        print(resultado['answer'])\\n",
                "\\n",
                "        if resultado['contexto_encontrado'] and resultado['citacoes']:\\n",
                "            print(\\\"\\\\nFontes consultadas:\\\")\\n",
                "            for citacao in resultado['citacoes']:\\n",
                "                print(f\\\"- {citacao}\\\")\\n",
                "        \\n",
                "        print(\\\"-\\\" * 50)\\n",
                "\\n",
                "# Para iniciar o chatbot, remova o comentário da linha abaixo e execute a célula.\\n",
                "# chatbot_lanchonete()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Parte 4: Tópicos Adicionais & Boas Práticas (Bônus)\\n",
                "\\n",
                "Parabéns por chegar até aqui! Agora que o pipeline de RAG está funcional, vamos explorar algumas ideias que estavam no seu notebook original e como podemos levá-las adiante."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 4.1 - O Experimento de Triagem (Structured Output)\\n",
                "\\n",
                "Seu notebook original continha um experimento interessante para **triagem** de mensagens, usando a biblioteca Pydantic para forçar o LLM a retornar uma saída em um formato JSON específico. Isso é chamado de **Structured Output** e é extremamente útil!\\n",
                "\\n",
                "Vamos manter seu código original aqui como um exemplo de caso de uso. Ele poderia ser o primeiro passo em um fluxo mais complexo: primeiro, um LLM classifica a mensagem do usuário (triagem) e, depois, decide se deve acionar o RAG, encaminhar para um humano ou outra ação.\\n",
                "\\n",
                "**Ponto de aprendizado:** No seu código original, o LLM era inicializado dentro da função `triagem`. Isso funciona, mas é ineficiente, pois cria um novo objeto a cada chamada. Uma boa prática seria inicializar o LLM uma vez e reutilizá-lo, como fizemos com o `llm` principal no início deste notebook."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "TRIAGEM_PROMPT = \\\"\\\"\\\"Você é o assistente virtual da Lanchonete Candango Lanches.\\n",
                "Sua função é classificar as mensagens dos clientes de acordo com a ação necessária e retornar SOMENTE um JSON com:\\\\n\\\\n\\n",
                "{\\\\n\\\\n\\n",
                "    \\\"decisao\\\": \\\"RESPONDER_INFO\\\" | \\\"ENCAMINHAR_ATENDENTE\\\" | \\\"PEDIDO_ESPECIAL\\\",\\\\n\\\\n\\n",
                "    \\\"urgencia\\\": \\\"BAIXA\\\" | \\\"MEDIA\\\" | \\\"ALTA\\\"\\\\n\\\\n\\n",
                "}\\\\n\\\\n\\n",
                "Regras:\\\\n\\\\n\\n",
                "- **RESPONDER_INFO**: Perguntas claras sobre o cardápio, horários, ingredientes (Ex: \\\"Qual o preço do X-Bacon?\\\")\\\\n\\\\n\\n",
                "- **ENCAMINHAR_ATENDENTE**: Mensagens complexas, reclamações, ou pedidos de informação muito específicos que não estão nos documentos (Ex: \\\"Tive um problema com meu pedido.\\\")\\\\n\\\\n\\n",
                "- **PEDIDO_ESPECIAL**: Solicitações fora do padrão do cardápio ou que exigem confirmação (Ex: \\\"Posso pedir um sanduíche sem cebola?\\\").\\\\n\\\\n\\n",
                "Analise a mensagem e decida a ação mais apropriada.\\\"\\\"\\\"\\n",
                "\\n",
                "class TriagemOut(BaseModel):\\n",
                "    decisao: Literal[\\\"RESPONDER_INFO\\\", \\\"ENCAMINHAR_ATENDENTE\\\", \\\"PEDIDO_ESPECIAL\\\"] = Field(\\n",
                "        description=\\\"Decisão da triagem.\\\"\\n",
                "    )\\n",
                "    urgencia: Literal[\\\"BAIXA\\\", \\\"MEDIA\\\", \\\"ALTA\\\"] = Field(\\n",
                "        description=\\\"Nível de urgência da solicitação.\\\"\\n",
                "    )\\n",
                "\\n",
                "# Mantendo a estrutura original do seu código para fins didáticos.\\n",
                "def triagem(mensagem: str) -> Dict:\\n",
                "  # Ponto de aprendizado: inicializar o LLM aqui é ineficiente.\\n",
                "  # O ideal seria reutilizar uma instância já criada.\\n",
                "  llm_triagem = ChatGoogleGenerativeAI(\\n",
                "      model=\\\"gemini-1.5-flash\\\",\\n",
                "      temperature=0.0, # Temperatura 0 para ser mais determinístico na classificação\\n",
                "      google_api_key=GOOGLE_API_KEY\\n",
                "  ).with_structured_output(TriagemOut)\\n",
                "\\n",
                "  saida: TriagemOut = llm_triagem.invoke([\\n",
                "      SystemMessage(content=TRIAGEM_PROMPT),\\n",
                "      HumanMessage(content=mensagem)\\n",
                "  ])\\n",
                "\\n",
                "  return saida.model_dump()\\n",
                "\\n",
                "# Testando a função de triagem\\n",
                "print(\\\"--- Testando o Experimento de Triagem ---\\\")\\n",
                "mensagem_teste = \\\"Tive um problema com a entrega, meu sanduíche veio frio.\\\"\\n",
                "resultado_triagem = triagem(mensagem_teste)\\n",
                "print(f\\\"Mensagem: '{mensagem_teste}'\\\")\\n",
                "print(f\\\"Resultado da triagem: {resultado_triagem}\\\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 4.2 - Sugestões de Inovação e Próximos Passos\\n",
                "\\n",
                "Excelente trabalho! Agora que você tem um pipeline de RAG funcional, o céu é o limite. Aqui estão algumas ideias para evoluir seu projeto:\\n",
                "\\n",
                "1.  **Usar LangGraph para Fluxos Complexos:** O experimento de triagem é um exemplo perfeito. E se, após a triagem, o sistema decidisse autonomamente se deve chamar o RAG ou fazer outra coisa? **LangGraph** é a ferramenta ideal para isso. Com ele, você pode construir fluxos de estado (grafos) onde cada nó é uma ação (triar, buscar no RAG, pedir mais informações ao usuário) e as arestas são as condições para ir de um estado para outro. É o próximo passo para criar agentes mais autônomos.\\n",
                "\\n",
                "2.  **Adicionar Memória:** Para que o chatbot se lembre de interações passadas na mesma conversa, você pode adicionar um componente de **memória** (disponível no LangChain). Isso permite que o usuário faça perguntas de acompanhamento sem precisar repetir o contexto.\\n",
                "\\n",
                "3.  **Avaliação Automática de Respostas:** Como saber se o seu RAG está melhorando? Você pode criar um sistema de avaliação (ex: usando o próprio LLM para comparar a resposta gerada com uma resposta \\\"padrão\\\") para medir a qualidade e a relevância das respostas. Frameworks como o LangSmith são ótimos para isso.\\n",
                "\\n",
                "4.  **Feedback Loop com Usuários:** Adicione uma forma simples de o usuário avaliar a resposta (👍/👎). Esse feedback é ouro! Ele pode ser usado para identificar respostas ruins e até para re-treinar ou ajustar seu sistema."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 4.3 - Dicas para Implementação e Produção\\n",
                "\\n",
                "Levar um projeto de IA do notebook para o mundo real requer algumas práticas importantes:\\n",
                "\\n",
                "1.  **Exportação do Chatbot:**\\n",
                "    *   **API com FastAPI ou Flask:** A maneira mais comum de colocar seu chatbot em produção é envolvê-lo em uma API. FastAPI é uma excelente escolha por sua performance e facilidade de uso.\\n",
                "    *   **Interface com Streamlit ou Gradio:** Para criar uma interface de demonstração rápida e interativa, bibliotecas como Streamlit e Gradio são perfeitas.\\n",
                "    *   **Docker:** Empacote sua aplicação e todas as suas dependências em um contêiner Docker. Isso garante que ela funcione de forma consistente em qualquer ambiente.\\n",
                "\\n",
                "2.  **Integração com Fontes Externas:**\\n",
                "    *   Seu Vector Store FAISS atual vive na memória. Para produção, você vai querer um banco de dados vetorial persistente e escalável. Algumas opções populares são **Pinecone**, **Weaviate** e **Postgres com a extensão `pgvector`**.\\n",
                "\\n",
                "3.  **Boas Práticas de Código:**\\n",
                "    *   **Modularização:** Em vez de um único notebook, separe seu código em módulos Python (`.py`). Por exemplo, um arquivo para o pipeline de RAG, outro para a configuração do LLM, e um `main.py` para a sua API FastAPI.\\n",
                "    *   **Gerenciamento de Segredos:** Nunca deixe chaves de API no código. Use variáveis de ambiente (com `python-dotenv`) ou um serviço de gerenciamento de segredos do seu provedor de nuvem (AWS Secrets Manager, Google Secret Manager)."
            ]
        }
    ],
    "metadata": {
        "colab": {
            "provenance": []
        },
        "kernelspec": {
            "display_name": "Python 3",
            "name": "python3"
        },
        "language_info": {
            "name": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open("notebook_refatorado.ipynb", "w", encoding='utf-8') as f:
    json.dump(notebook, f, indent=2, ensure_ascii=False)

print("Notebook 'notebook_refatorado.ipynb' foi criado com sucesso.")
