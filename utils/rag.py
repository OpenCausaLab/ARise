import json
from functools import cache

import nltk
from retriv import SparseRetriever

nltk.download = lambda *args, **kwargs: None

_retriever_cache = {}


@cache
def get_retriever(data_path, data_idx):
    """Get retriever from database"""
    if (data_path, data_idx) in _retriever_cache:
        return _retriever_cache[(data_path, data_idx)]

    try:
        # Initialize retriever
        retriever = SparseRetriever()

        with open(data_path) as f:
            database = json.load(f)
            context = database[data_idx]["context"]

        # Create documents
        documents = []
        idx = 0
        for ele in context:
            theme = ele[0]
            infos = ele[1]
            for info in infos:
                documents.append(
                    {
                        "id": idx,
                        "text": f"keyword: {theme}\nfacts: {info}",
                        "keyword": theme,
                        "facts": info,
                    }
                )
                idx += 1

        # with open(data_path) as f:
        #     dataraw = json.load(f)
        #     context = dataraw[data_idx]["context"]

        # documents = []
        # idx = 0
        # for ele in context:
        #     keyword = list(ele.keys())[0]
        #     facts = ele[keyword]
        #     documents.append(
        #         {
        #             "id": idx,
        #             "text": f"keyword: {keyword}\nfacts: {facts}",
        #             "keyword": keyword,
        #             "facts": facts,
        #         }
        #     )
        #     idx += 1

        # with open(data_path) as f:
        #     dataraw = json.load(f)
        #     context = dataraw[data_idx]["context"]

        # documents = []
        # idx = 0
        # for ele in context:
        #     keyword = list(ele.keys())[0]
        #     facts = ele[keyword]
        #     i = 0
        #     while i < len(facts):
        #         while len(facts[i]) < 50 and i + 1 < len(facts):
        #             facts[i] = facts[i] + " " + facts[i + 1]
        #             del facts[i + 1]
        #         i += 1
        #     for fact in facts:
        #         documents.append(
        #             {
        #                 "id": idx,
        #                 "text": f"keyword: {keyword}\nfacts: {fact}",
        #                 "keyword": keyword,
        #                 "facts": fact,
        #             }
        #         )
        #         idx += 1
        # Build index in memory
        retriever.index(documents)

        _retriever_cache[(data_path, data_idx)] = retriever
        return retriever

    except Exception as e:
        print(f"Error initializing retriever: {e}")
        return None


def retrieve(query, data_path, data_idx, topk=2):
    """
    Retrieve similar math problems and their relevance scores

    Returns:
        str: Concatenated results from the retrieved documents
    """
    try:
        print("Starting retrieval process...")

        retriever = get_retriever(data_path, data_idx)
        if retriever is None:
            print("Failed to initialize retriever")
            return ""

        print("Retriever initialized, starting search...")
        docs = retriever.search(query, cutoff=topk)
        print(f"Search completed, found {len(docs)} documents")

        results = []
        for doc in docs:
            results.append(
                f"{doc['keyword']}".strip() + ": " + f"{doc['facts']}".strip()
            )

        return "\n".join(results)

    except Exception as e:
        print(f"Error during retrieval: {str(e)}")
        return ""
