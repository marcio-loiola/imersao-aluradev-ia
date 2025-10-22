# Chatbot com IA Generativa e RAG para a Lanchonete Candango Lanches

## 🚀 Resumo do Projeto

Este repositório contém um projeto de **Inteligência Artificial Generativa** que desenvolve um chatbot inteligente para a "Candango Lanches". A solução utiliza a arquitetura **RAG (Retrieval-Augmented Generation)** para responder a perguntas de clientes e funcionários com base em uma base de conhecimento interna (documentos de políticas da empresa).

O objetivo é demonstrar a aplicação prática de Grandes Modelos de Linguagem (LLMs) em um cenário de negócios, cobrindo desde a engenharia de dados (processamento de documentos) até a implementação de um sistema de IA funcional e a visão de produto para sua evolução.

---

## 🎯 Capacidades e Features

Este projeto demonstra competências em três áreas-chave:

### 🧠 Inteligência Artificial
- **Pipeline de RAG Completo:** Implementação de um fluxo de ponta a ponta que busca informações em documentos e gera respostas contextuais, utilizando LangChain.
- **Uso de LLMs (Google Gemini):** Integração com o modelo `gemini-1.5-flash` para geração de texto e embeddings.
- **Busca Vetorial Semântica:** Conversão de documentos em vetores (embeddings) e armazenamento em um Vector Store (FAISS) para buscas de similaridade eficientes.
- **Saída Estruturada (Structured Output):** Um experimento bônus demonstra como usar LLMs para tarefas de classificação (triagem de mensagens), forçando a saída em formato JSON, uma técnica crucial para criar agentes de IA mais robustos.

### 📊 Dados
- **Engenharia de Dados:** Carregamento e processamento de documentos não estruturados (PDFs) para criar uma base de conhecimento.
- **Text Chunking Estratégico:** Divisão inteligente de documentos em pedaços menores (`chunks`) para otimizar a busca e se adequar aos limites de contexto dos LLMs.
- **Criação de Base de Conhecimento:** Geração programática de PDFs para simular um ambiente real onde a base de conhecimento pode ser facilmente atualizada.

### 📈 Negócios
- **Visão de Produto:** O projeto vai além da implementação, sugerindo próximos passos como o uso de **LangGraph** para criar agentes autônomos, adição de **memória conversacional** e implementação de **ciclos de feedback** (avaliação de respostas), demonstrando uma mentalidade focada na evolução e melhoria contínua do produto.
- **Eficiência Operacional:** A solução automatiza o atendimento a perguntas frequentes, liberando a equipe para focar em tarefas mais complexas.
- **Escalabilidade:** As boas práticas sugeridas (APIs, Docker, bancos de dados vetoriais persistentes) mostram um caminho claro para levar a solução do protótipo à produção.

---

## 🏗️ Arquitetura da Solução (Pipeline de RAG)

O fluxo de funcionamento do chatbot pode ser resumido nos seguintes passos:

1.  **Carregamento e Divisão:** Documentos PDF com as políticas da lanchonete são carregados e divididos em `chunks`.
2.  **Criação de Embeddings:** Cada `chunk` é transformado em um vetor numérico (embedding) usando o modelo `models/embedding-001` do Google.
3.  **Armazenamento Vetorial:** Os embeddings são armazenados no **FAISS**, um banco de dados vetorial em memória.
4.  **Recuperação (Retrieval):** Quando um usuário faz uma pergunta, o sistema a converte em um embedding e busca os `chunks` mais relevantes (semanticamente similares) no FAISS.
5.  **Geração Aumentada (Generation):** Os `chunks` recuperados (contexto) e a pergunta original são enviados ao LLM (Gemini), que gera uma resposta final baseada nas informações fornecidas.

---

## 📂 Estrutura do Repositório

-   `README.md`: Visão geral do projeto, arquitetura e instruções de uso.
-   `requirements.txt`: Lista de todas as bibliotecas Python necessárias para executar o projeto.
-   `build_notebook.py`: **Arquivo-fonte principal.** Este script Python gera o notebook final (`notebook_refatorado.ipynb`). Se você quiser fazer alterações ou melhorias, edite este arquivo.
-   `notebook_refatorado.ipynb`: O Jupyter Notebook gerado, pronto para ser executado no Google Colab ou localmente. **Não edite este arquivo diretamente**, pois suas alterações serão perdidas na próxima vez que o script de build for executado.

---

## 🛠️ Tecnologias Utilizadas

- **Linguagem:** Python
- **Orquestração de LLMs:** LangChain
- **Modelo de Linguagem (LLM):** Google Gemini 1.5 Flash
- **Embeddings:** Google `embedding-001`
- **Vector Store:** FAISS (Facebook AI Similarity Search)
- **Manipulação de Documentos:** PyMuPDF, RecursiveCharacterTextSplitter
- **Notebook:** Jupyter/Google Colab

---

## ⚙️ Como Executar o Projeto

1.  **Clone o repositório:**
    ```bash
    git clone <URL_DO_REPOSITORIO>
    cd <NOME_DA_PASTA>
    ```

2.  **Instale as dependências:**
    Certifique-se de ter o Python 3.8+ instalado e execute:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure sua API Key:**
    - Obtenha uma chave de API do **Google AI Studio**.
    - Se estiver usando o Google Colab (recomendado), crie um segredo chamado `GEMINI_API_KEY` e cole sua chave lá. O notebook está configurado para ler a chave a partir dos segredos do Colab.
    - Se estiver rodando localmente, você pode adaptar o código para ler a chave de uma variável de ambiente.

4.  **Execute o Notebook:**
    - Abra o arquivo `notebook_refatorado.ipynb` no Jupyter ou Google Colab.
    - Execute as células em ordem. O notebook é autoexplicativo e irá:
        - Criar os arquivos PDF de exemplo.
        - Construir o pipeline de RAG.
        - Iniciar um chatbot interativo no final para você testar.

5.  **Modificando o Notebook (Avançado):**
    - O arquivo `notebook_refatorado.ipynb` é gerado automaticamente pelo script `build_notebook.py`.
    - Se desejar fazer alterações permanentes, edite o script `build_notebook.py` e execute-o (`python build_notebook.py`) para gerar uma nova versão do notebook.
