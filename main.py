from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import numpy as np
from time import sleep
import pandas as pd
import requests
import builtins
from dotenv import load_dotenv
import os

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "sk-b1uxjjuQ9afaEmJH02UUNQ")
LLM_API_KEY = os.getenv("LLM_API_KEY", "sk-aq0mXtn7nlW4aBnAfUocSQ")
EMBEDDER_API_KEY = os.getenv("EMBEDDER_API_KEY", "sk-b1uxjjuQ9afaEmJH02UUNQ")
MAX_RETRIES_COUNTER = 5

def retrieve_context(query: str, k: int = 10):
        """Retrieve information to help answer a query."""
        retrieved_docs = vectorstore.similarity_search(query, k=k)

        indexes = set()

        for doc in retrieved_docs:
            for idx in doc.metadata["indexes"]:
                indexes.add(idx)

        serialized = "\n\n".join(
            (f"Source: {doc.metadata}\nContent: {doc.page_content}")
            for doc in retrieved_docs
        )
        return serialized, retrieved_docs, indexes

def rerank_docs(query, documents, key=EMBEDDER_API_KEY):
    # Базовый url - сохранять без изменения
    url = "https://ai-for-finance-hack.up.railway.app/rerank"
    # Формируем заголовок для запроса
    headers = {
        # Указываем тип получаемого контента
        "Content-Type": "application/json",
        # Указываем наш ключ, полученный ранее
        "Authorization": f"Bearer {key}"
    }
    # Формируем сам запрос
    payload = {
        # Указываем модель
        "model": "deepinfra/Qwen/Qwen3-Reranker-4B",
        # Формируем текст запроса
        "query": query,
        # Добавляем документы для ранжирования
        "documents": documents
    }
    # Отправляем запрос
    response = requests.post(url, headers=headers, json=payload)
    # Возвращаем результат запроса

    return response.json()

counter = 0

def get_context_question_pair(question, top_for_retriever=20,top_for_reranker=2):
    global counter

    context = retrieve_context(question, top_for_retriever)
    sleep(1)
    res = [docs_segments[idx] for idx in context[2]]

    answer = rerank_docs(
        query=question,
        documents=res,
        key=EMBEDDER_API_KEY
        )
    
    for i in range(MAX_RETRIES_COUNTER):
        if 'results' in answer:
            break
        elif i != MAX_RETRIES_COUNTER - 1:
            #print("reranker model request retry")
            sleep(0.5)
        else:
            raise Exception("Reranker model is inaccessable")
        
        answer = rerank_docs(
            query=question,
            documents=res,
            key=EMBEDDER_API_KEY
        )

    sorted_queries = builtins.sorted(answer['results'], key = lambda x: x['relevance_score'],reverse=True)
    final_indexes = []

    symbol_counter = 0
    for dict in sorted_queries[:top_for_reranker]:
      final_indexes.append(dict['index'])
      symbol_counter += len(res[dict['index']])
      if symbol_counter > 10000:
          break

    res_reranked = [res[idx] for idx in final_indexes]

    #print(counter)
    counter += 1

    return (question, "Document: " + "\n Document:".join(res_reranked))

if __name__ == "__main__":
    data_loader = CSVLoader(file_path='train_data.csv', encoding="utf-8", content_columns=["text"])
    docs = data_loader.load()

    chunks = []
    docs_segments = []
    total_length = 0
    j = 0

    for doc in docs:
        doc_segments = doc.page_content.replace("###", "--").split("##")
        if doc_segments[0] == "text: ":
            # get rid of default text segment
            doc_segments = doc_segments[1:]

        docs_segments.extend(doc_segments)

        total_length += len(doc_segments)


        for i in range(max(1,len(doc_segments))):
            chunk_len = len(doc_segments[i:i+1])
            chunks.append((doc_segments[i:i+1], [j]))
            j += 1

    chunks_docs = []

    for i, (text, indexes) in enumerate(chunks):
        chunks_docs.append(Document("\n".join(text), metadata={"indexes": indexes}))

    embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            base_url="https://ai-for-finance-hack.up.railway.app/",
            api_key=EMBEDDER_API_KEY
            )

    ######

    vectorstore = FAISS.from_documents(chunks_docs[:1], embeddings)

    batch_size = 100

    chunks_docs = chunks_docs[1:]

    for i in range(len(chunks_docs) //  batch_size + 1):
        if i * batch_size >= len(chunks_docs):
            break
        vectorstore.add_documents(documents=chunks_docs[i * batch_size: min((i + 1) * batch_size,len(chunks_docs)-1)])
        sleep(2)
        #print(i * batch_size, (i + 1) * batch_size)

    ######


    ######

    #vectorstore = FAISS.load_local(
    #    "faiss_index", 
    #    embeddings, 
    #    allow_dangerous_deserialization=True
    #)

    ######

    #print(1)

    #vectorstore.save_local("faiss_index") # !!!!!!!!!!!!!!!!!!!!!!!

    questions_loader = CSVLoader(file_path='questions.csv', encoding="utf-8", content_columns=["Вопрос"])
    questions_docs = questions_loader.load()

    questions = [doc.page_content[8:] for doc in questions_docs]

    ######

    queries = [get_context_question_pair(question, top_for_retriever=15, top_for_reranker=5) for question in questions]

    ######


    ######

    #context = pd.read_csv('context.csv')
    #context = context["text"].values

    #queries = [(context[i], question) for i, question in enumerate(questions)]

    ######

    #print(2)

    llm_model = ChatOpenAI(
        model="openrouter/google/gemma-3-27b-it",
        base_url="https://ai-for-finance-hack.up.railway.app/",
        api_key="sk-aq0mXtn7nlW4aBnAfUocSQ",
        temperature=0.4,
        ).with_retry(
        stop_after_attempt=6,
        )

    chain = (
        {"query": lambda x: x[0]}
        | {"context": lambda x: x[1]}
        | ChatPromptTemplate.from_template("""Ты - умный и полезный ассистент для консультирования клиентов банка. 
                                                 Твоя задача - давать развернутые ответы на вопросы клиента банка.
                                                 Для помощи в ответе тебе будет выдана контекстная информация.
                                                 При ответе учитывай информацию из контекста.
                                                 Если информации в контексте окажется недостаточно,
                                                 дай клиенту общую информацию из контекста и порекомендуй обратиться к юристу или финансовому специалисту для конкретного случая.
                                                 Не пиши нашел ли ты информацию в контексте или  нет. Для клиента это должно остаться тайной.
                                                 Ответ должен быть подробным, с заголовками, если можно что-то сравнить, то делай сравнения в формате таблиц.
                                                 Ответ напиши на русском языке.
                                                 Вот пример ответа:

                                                 Контекст: {context}
                                                 {query}""")
        | llm_model
        | StrOutputParser()
    )

    response = chain.batch(queries, {"max_concurrency": 25})

    data = {
        "ID вопроса": np.arange(1, len(questions) + 1),
        "Вопрос": questions,
        "Ответ на вопрос": list(response),
    }

    submission = pd.DataFrame(data)
    submission.to_csv('submission.csv', index=False)
