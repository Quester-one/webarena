if __name__=="__main__":
    import os
    from config_private import http_proxy, https_proxy
    os.environ["http_proxy"] = http_proxy
    os.environ["https_proxy"] = https_proxy
    from sentence_transformers import SentenceTransformer, util

    # 加载SBERT模型
    model_name = 'paraphrase-MiniLM-L6-v2'  # 选择合适的SBERT模型
    model = SentenceTransformer(model_name)

    # 定义一些示例文本
    query = "自然语言处理在人工智能领域中的应用"
    documents = [
        "自然语言处理是人工智能的一个重要领域。",
        "人工智能包括机器学习、深度学习和自然语言处理等方面。",
        "计算机科学家使用自然语言处理来改善机器翻译。",
        "深度学习在自然语言处理任务中取得了显著的成果。",
    ]

    # 将文本转换为嵌入向量
    query_embedding = model.encode(query, convert_to_tensor=True)
    document_embeddings = model.encode(documents, convert_to_tensor=True)

    # 计算相似度
    cosine_scores = util.pytorch_cos_sim(query_embedding, document_embeddings)[0]

    # 打印相似度分数和对应的文本
    for i, score in enumerate(cosine_scores):
        print(f"相似度分数: {score:.4f}, 文本: {documents[i]}")
