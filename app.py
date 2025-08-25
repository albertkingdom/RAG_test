import os
import bs4
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Qdrant
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain import hub
from langchain.schema import HumanMessage, SystemMessage
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors import FlashrankRerank
import langchain

# langchain.debug = True


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
# BASE_URL="https://litellm-ekkks8gsocw.dgx-coolify.apmic.ai"
# MODEL="gemma-3-12b-it"

# 使用開源model
# llm = ChatOpenAI(model_name=MODEL, temperature=0, base_url=BASE_URL, api_key=APMIC_API_KEY)

# 使用OpenAI
llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

# 1. 使用 MultiQueryRetriever 作為基礎檢索器
base_retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(),
    llm=llm
)

# 2. 使用 FlashrankRerank 對召回的文件進行重新排序，篩選出最相關的3個
compressor = FlashrankRerank(top_n=3)
retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever
)

# 範例查詢
question = "此網頁的主要內容是什麼？"

print("--- 正在檢索文件 (Re-ranking前) ---")
docs_before = base_retriever.invoke(question)
for i, doc in enumerate(docs_before):
    cleaned_content = doc.page_content[:150].replace('\n', ' ')
    print(f"Rank {i+1}: {cleaned_content}...")
print("\n" + "="*30 + "\n")

print("--- 正在檢索文件 (Re-ranking後) ---")
docs_after = retriever.invoke(question)
for i, doc in enumerate(docs_after):
    cleaned_content = doc.page_content[:150].replace('\n', ' ')
    print(f"Rank {i+1}: {cleaned_content}...")
print("\n" + "="*30 + "\n")


# The rest of the logic uses the re-ranked documents
combined = "\n\n".join(d.page_content for d in docs_after)
# print(combined + "\n\n")
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content=f"Context:\n{combined}\n\nQuestion: {question}"),
]
response = llm.invoke(messages)
print("--- LLM 的最終回覆 ---")
print(response.content)

