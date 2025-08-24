import os
import bs4
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Qdrant
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain import hub
from langchain.schema import HumanMessage, SystemMessage
from langchain.retrievers.multi_query import MultiQueryRetriever
import langchain

langchain.debug = True


# ====== API Key is loaded from .env file by docker-compose ======

# ====== Step 1: 載入與切割文件 ======
loader = WebBaseLoader(
    web_path=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    )
)

docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# ====== Step 2: 建立 Qdrant Vector DB ======
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

vectorstore = Qdrant.from_documents(
    documents=splits,
    embedding=embeddings,
    url="http://qdrant:6333",   # 你的 Qdrant REST API URL
    collection_name="rag_demo"
)

# ====== Step 3: 檢索 + 生成 (優化: MultiQueryRetriever) ======
# prompt = hub.pull("rlm/rag-prompt")  # 從 LangChain Hub 取得現成 prompt
BASE_URL="https://litellm-ekkks8gsocw.dgx-coolify.apmic.ai"
MODEL="gemma-3-12b-it"

# 使用開源model
# llm = ChatOpenAI(model_name=MODEL, temperature=0, base_url=BASE_URL, api_key=APMIC_API_KEY)

# 使用OpenAI
llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

# 1. 使用 MultiQueryRetriever 優化檢索， LLM 根據使用者的原始問題，生成多個不同角度、不同措辭的相似問題。然後用這些問題一起去資料庫中搜尋
retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(),
    llm=llm
)

# 範例查詢
question = "此網頁的主要內容是什麼？"
docs = retriever.invoke(question)
combined = "\n\n".join(d.page_content for d in docs)
# print(combined + "\n\n")
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content=f"Context:\n{combined}\n\nQuestion: {question}"),
]
response = llm.invoke(messages)
print(response.content)
