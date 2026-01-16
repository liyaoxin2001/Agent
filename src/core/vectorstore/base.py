"""
向量存储基础接口定义

向量存储模块负责管理和检索高维向量数据，是 RAG 系统中的关键组件。

核心功能：
- 存储文档向量到向量数据库
- 基于相似度进行向量检索
- 支持向量的持久化存储
- 提供高效的相似度搜索接口

架构设计：
- BaseVectorStore: 抽象接口层，支持多种向量存储实现
- FAISSVectorStore: 基于 FAISS 的具体实现

使用场景：
- RAG 系统中的文档检索
- 语义搜索
- 推荐系统
- 文本聚类

TODO: 你需要实现这个接口的具体类
"""

from abc import ABC, abstractmethod
from typing import List, Optional
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
import os
import dotenv
from langchain.embeddings.base import Embeddings
from langchain_openai import OpenAIEmbeddings

from src.core.embedding.base import BaseEmbedding

dotenv.load_dotenv()

class BaseVectorStore(ABC):
    """
    向量存储基础接口

    定义了向量存储系统应该提供的核心功能接口。
    通过抽象接口设计，支持不同的向量存储实现（如 FAISS、Chroma、Milvus 等）。

    设计模式：
    - 策略模式：允许在运行时切换不同的向量存储实现
    - 抽象工厂：定义统一的接口规范

    核心职责：
    1. 文档向量化存储
    2. 相似度检索
    3. 数据持久化
    4. 文档管理（添加/删除）
    """

    def __init__(self,
                 embedding:BaseEmbedding,
                 persist_directory: Optional[str] = None):
        """
        初始化向量存储

        Args:
            persist_directory: 向量库持久化保存的目录路径
                            如果为 None，则不进行持久化存储
                            如果指定路径，会自动创建目录

        初始化逻辑：
        1. 设置持久化目录路径
        2. 初始化向量存储相关的配置
        3. 准备 Embedding 模型（由子类实现）
        """
        self.embedding = embedding
        self.persist_directory = persist_directory
    
    @abstractmethod
    def add_documents(
        self,
        documents: List[Document],
        embeddings: Optional[List[List[float]]] = None
    ):
        """
        添加文档到向量库

        支持两种模式：
        1. 自动计算向量：传入文档列表，内部自动计算向量
        2. 预计算向量：传入文档列表和预计算的向量，提高效率

        Args:
            documents: LangChain Document 对象列表
                      每个 Document 包含 page_content（文本内容）和 metadata（元数据）
            embeddings: 预计算的向量列表（可选）
                       - 如果提供：使用这些向量，避免重复计算
                       - 如果不提供：自动调用 Embedding 模型计算向量
                       - 维度必须与 Embedding 模型输出一致

        Raises:
            Exception: 当添加文档失败时抛出异常

        示例：
            # 自动计算向量
            docs = [Document(page_content="Python 教程")]
            vectorstore.add_documents(docs)

            # 预计算向量（更高效）
            vectors = embedding.embed_documents([doc.page_content for doc in docs])
            vectorstore.add_documents(docs, vectors)
        """
        pass
    
    @abstractmethod
    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict] = None
    ) -> List[Document]:
        """
        基于相似度的文本搜索

        工作流程：
        1. 将查询文本向量化
        2. 在向量空间中查找最相似的向量
        3. 返回对应的文档，按相似度排序

        Args:
            query: 查询文本字符串
                  将被转换为向量进行相似度计算
            k: 返回的最相似文档数量，默认 4 个
              数量太多可能包含不相关结果
              数量太少可能遗漏有用信息
            filter: 过滤条件（可选）
                   用于在搜索前过滤文档
                   具体支持的过滤条件取决于向量存储实现

        Returns:
            List[Document]: 相似度最高的文档列表
                          按相似度降序排列（最相关的排在前面）
                          每个 Document 包含 page_content 和 metadata

        Raises:
            Exception: 当搜索失败时抛出异常

        示例：
            # 基本搜索
            results = vectorstore.similarity_search("什么是 Python？", k=3)

            # 带过滤的搜索（如果支持）
            results = vectorstore.similarity_search(
                "什么是 Python？",
                k=3,
                filter={"source": "tutorial.pdf"}
            )
        """
        pass
    
    @abstractmethod
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict] = None
    ) -> List[tuple[Document, float]]:
        """
        带相似度分数的搜索

        除了返回文档外，还返回每个文档的相似度分数。
        分数表示文档与查询的相关程度，分数越高越相关。

        相似度分数的含义：
        - FAISS: 距离分数，越小越相似（L2 距离）
        - 其他: 相似度分数，越大越相似（余弦相似度）

        Args:
            query: 查询文本字符串
            k: 返回的文档数量
            filter: 过滤条件（可选）

        Returns:
            List[Tuple[Document, float]]: (文档, 分数) 元组列表
                                         按分数排序（最相关的排在前面）
                                         分数含义取决于具体的向量存储实现

        Raises:
            Exception: 当搜索失败时抛出异常

        示例：
            results = vectorstore.similarity_search_with_score("Python 教程", k=2)
            # 返回: [(Document("Python 基础教程"), 0.85), (Document("Python 进阶"), 0.72)]

            # 分析分数
            for doc, score in results:
                print(f"文档: {doc.page_content[:50]}...")
                print(f"相似度分数: {score}")
        """
        pass
    
    @abstractmethod
    def delete(self, ids: Optional[List[str]] = None):
        """
        删除文档

        不同的向量存储对删除操作的支持程度不同：
        - FAISS: 不支持按 ID 删除，只能清空整个向量库
        - Chroma/Milvus: 支持按 ID 删除特定文档

        Args:
            ids: 要删除的文档 ID 列表
                - 如果为 None：清空整个向量库
                - 如果为列表：删除指定的文档（如果支持）

        Raises:
            NotImplementedError: 如果向量存储不支持按 ID 删除
            Exception: 当删除操作失败时抛出

        示例：
            # 清空整个向量库
            vectorstore.delete()

            # 删除特定文档（如果支持）
            vectorstore.delete(ids=["doc1", "doc2"])
        """
        pass

    @abstractmethod
    def persist(self):
        """
        持久化向量库到磁盘

        将内存中的向量数据保存到磁盘文件中，包括：
        - 向量数据
        - 索引结构
        - 文档存储
        - 相关元数据

        持久化后，可以通过 load 操作恢复向量库状态。

        Raises:
            Exception: 当持久化失败时抛出

        示例：
            # 设置持久化目录
            vectorstore = FAISSVectorStore(persist_directory="./data/vectors")

            # 添加文档后自动持久化
            vectorstore.add_documents(documents)
            vectorstore.persist()  # 保存到磁盘

            # 下次启动时可以加载
            vectorstore2 = FAISSVectorStore(persist_directory="./data/vectors")
            # 自动加载已保存的向量库
        """
        pass


# TODO: 实现 FAISSVectorStore 类
# 提示：
# 1. 继承 BaseVectorStore
# 2. 使用 langchain_community.vectorstores.FAISS
# 3. 实现所有抽象方法
# 4. 参考文档: https://python.langchain.com/docs/integrations/vectorstores/faiss
# 
# 关键步骤：
# - 使用 FAISS.from_documents() 创建向量库
# - 使用 save_local() 持久化
# - 使用 load_local() 加载
# - 使用 similarity_search() 检索
class LangChainEmbeddingAdapter(Embeddings):
    def __init__(self, embedding: BaseEmbedding):
        self.embedding = embedding

    def embed_documents(self, texts):
        return self.embedding.embed_documents(texts)

    def embed_query(self, text):
        return self.embedding.embed_query(text)

class FAISSVectorStore(BaseVectorStore):
    """
    基于 FAISS 的向量存储实现

    FAISS (Facebook AI Similarity Search) 是一个高效的相似度搜索库，
    特别适用于大规模向量数据的快速检索。

    核心特性：
    - 支持十亿级向量的快速搜索
    - 多种索引算法（L2、余弦相似度、内积）
    - GPU 加速支持
    - 内存效率高

    工作原理：
    1. 将文档向量化后存储在 FAISS 索引中
    2. 使用近似最近邻搜索算法进行相似度检索
    3. 支持精确搜索和近似搜索的平衡

    限制：
    - 不支持按 ID 删除特定文档
    - 索引重建需要重新添加所有文档

    使用场景：
    - 大规模文档检索
    - 实时相似度搜索
    - 需要高性能的向量搜索应用
    """

    def __init__(
            self,
            embedding: BaseEmbedding,
            persist_directory: Optional[str] = None
    ):
        super().__init__(embedding, persist_directory)

        self.lc_embedding = LangChainEmbeddingAdapter(embedding)
        self.vectorstore: Optional[FAISS] = None

        if persist_directory and os.path.exists(persist_directory):
            try:
                self.vectorstore = FAISS.load_local(
                    persist_directory,
                    self.lc_embedding,
                    allow_dangerous_deserialization=True
                )
            except Exception:
                self.vectorstore = None

    def add_documents(self, documents: List[Document], embeddings: Optional[List[List[float]]] = None):
        """
        添加文档到 FAISS 向量库

        支持两种添加模式：
        1. 首次创建向量库：当 self.vectorstore 为 None 时
        2. 增量添加：添加到现有向量库中

        两种向量化模式：
        1. 自动计算：传入文档，内部调用 Embedding 模型
        2. 预计算：传入文档和向量，避免重复计算

        Args:
            documents: LangChain Document 对象列表
            embeddings: 预计算的向量列表（可选）

        Raises:
            Exception: 当添加失败时抛出异常

        实现细节：
        - from_documents(): 创建新向量库，自动计算向量
        - from_embeddings(): 使用预计算向量创建向量库
        - add_documents(): 增量添加，自动计算向量
        - add_embeddings(): 增量添加，使用预计算向量
        """
        try:
            # 检查向量库是否存在
            if self.vectorstore is None:
                # ===== 首次创建向量库 =====

                if embeddings is not None:
                    # 模式1: 使用预计算向量创建向量库
                    # 将文本和向量配对
                    text_embeddings = list(zip(
                        [doc.page_content for doc in documents],
                        embeddings
                    ))
                    # FAISS.from_embeddings() 创建向量库
                    # text_embeddings: [(text, vector), ...] 格式
                    # embedding: Embedding 模型（用于配置向量库）
                    self.vectorstore = FAISS.from_embeddings(
                        text_embeddings=text_embeddings,
                        embedding=self.lc_embedding  # ✅ 修复：使用 lc_embedding
                    )
                else:
                    # 模式2: 自动计算向量创建向量库
                    # FAISS.from_documents() 会自动调用 Embedding 模型
                    self.vectorstore = FAISS.from_documents(
                        documents=documents,
                        embedding=self.lc_embedding  # ✅ 修复：使用 lc_embedding
                    )
            else:
                # ===== 增量添加到现有向量库 =====

                if embeddings is not None:
                    # 使用预计算向量增量添加
                    text_embeddings = list(zip(
                        [doc.page_content for doc in documents],
                        embeddings
                    ))
                    # add_embeddings() 方法增量添加向量
                    self.vectorstore.add_embeddings(text_embeddings)
                else:
                    # 自动计算向量增量添加
                    # add_documents() 方法会自动计算向量
                    self.vectorstore.add_documents(documents)

        except Exception as e:
            raise Exception(f"添加文档失败: {str(e)}") from e

    def similarity_search(self, query: str, k: int = 4, filter: Optional[dict] = None) -> List[Document]:
        """
        FAISS 相似度搜索

        搜索流程：
        1. 检查向量库是否存在
        2. 将查询文本向量化（自动调用 Embedding 模型）
        3. 在 FAISS 索引中进行相似度搜索
        4. 返回最相似的文档列表

        FAISS 搜索特性：
        - 使用 L2 距离（欧几里得距离）作为相似度度量
        - 距离越小，相似度越高
        - 支持近似最近邻搜索，性能优异

        Args:
            query: 查询文本
            k: 返回文档数量
            filter: FAISS 不支持过滤条件，忽略此参数

        Returns:
            List[Document]: 相似度最高的文档列表，按相似度降序排列

        Raises:
            Exception: 当搜索失败时抛出异常
        """
        # 检查向量库是否存在
        if self.vectorstore is None:
            return []

        try:
            # 调用 FAISS 的 similarity_search 方法
            # 该方法会自动：
            # 1. 将 query 向量化
            # 2. 在索引中搜索最相似的 k 个向量
            # 3. 返回对应的文档
            docs = self.vectorstore.similarity_search(
                query=query,
                k=k
            )
            return docs
        except Exception as e:
            raise Exception(f"相似度搜索失败: {str(e)}") from e

    def similarity_search_with_score(self, query: str, k: int = 4, filter: Optional[dict] = None) -> List[tuple[Document, float]]:
        """
        FAISS 带相似度分数的搜索

        返回格式：[(Document, score), ...]
        分数的含义：
        - FAISS 使用 L2 距离，越小越相似
        - 分数 = 0 表示完全匹配
        - 分数越大，相似度越低

        应用场景：
        - 需要根据相似度阈值过滤结果
        - 调试搜索结果的质量
        - 实现更复杂的排序逻辑

        Args:
            query: 查询文本
            k: 返回文档数量
            filter: FAISS 不支持过滤条件

        Returns:
            List[Tuple[Document, float]]: (文档, 距离分数) 元组列表
                                        按距离升序排列（最相似的排在前面）

        Raises:
            Exception: 当搜索失败时抛出异常

        示例：
            results = vectorstore.similarity_search_with_score("Python", k=3)
            # [(Document("Python 教程"), 0.12), (Document("Java 教程"), 0.85), ...]

            # 过滤低相似度结果
            threshold = 0.5
            filtered = [(doc, score) for doc, score in results if score < threshold]
        """
        if self.vectorstore is None:
            return []

        try:
            # similarity_search_with_score 返回 (文档, 距离分数) 元组列表
            # 距离分数：L2 距离，越小越相似
            docs_with_scores = self.vectorstore.similarity_search_with_score(
                query, k=k
            )
            return docs_with_scores
        except Exception as e:
            raise Exception(f"带分数搜索失败: {str(e)}") from e

    def delete(self, ids: Optional[List[str]] = None):
        """
        删除文档（FAISS 限制实现）

        FAISS 的删除限制：
        - FAISS 索引不支持按 ID 删除特定向量
        - 删除操作需要重建整个索引
        - 生产环境中建议使用支持删除的向量存储（如 Chroma、Milvus）

        当前实现：
        - ids=None: 清空整个向量库（设置 self.vectorstore = None）
        - ids!=None: 抛出 NotImplementedError

        Args:
            ids: 文档 ID 列表（当前不支持，传入会报错）

        Raises:
            NotImplementedError: 当尝试删除特定文档时
            Exception: 当删除操作失败时

        替代方案：
        1. 重建向量库：删除不需要的文档后重新添加
        2. 使用其他向量存储：Chroma、Milvus 支持增删操作
        """
        try:
            if ids is None:
                # 清空整个向量库
                # 将向量库实例设为 None，释放内存
                self.vectorstore = None
                print("✅ 向量库已清空")
            else:
                # FAISS 不支持按 ID 删除特定文档
                raise NotImplementedError(
                    "FAISS 不支持按 ID 删除文档。如需删除特定文档，"
                    "建议使用 Chroma 或 Milvus 等支持增删操作的向量存储。"
                )
        except Exception as e:
            raise Exception(f"文档删除失败: {str(e)}") from e

    def persist(self):
        """
        持久化 FAISS 向量库到磁盘

        保存内容：
        - FAISS 索引文件（.faiss）
        - 向量数据文件（.pkl）
        - 文档存储数据
        - Embedding 模型配置

        文件结构：
        persist_directory/
        ├── index.faiss    # FAISS 索引文件
        └── index.pkl      # 向量和文档数据

        加载方式：
        - 构造函数会自动检测并加载现有向量库
        - FAISS.load_local() 方法加载文件

        Raises:
            Exception: 当持久化失败时抛出异常

        注意事项：
        - 保存的文件可能较大（取决于向量数量）
        - allow_dangerous_deserialization=True 在加载时需要
        """
        # 检查向量库和持久化目录
        if self.vectorstore is None:
            print("⚠️  向量库为空，无需持久化")
            return

        if not self.persist_directory:
            print("⚠️  未设置持久化目录，跳过持久化")
            return

        try:
            # 创建持久化目录（如果不存在）
            os.makedirs(self.persist_directory, exist_ok=True)

            # 调用 FAISS 的 save_local 方法
            # 保存索引和向量数据到指定目录
            self.vectorstore.save_local(self.persist_directory)
            print(f"✅ 向量库已保存到: {self.persist_directory}")

        except Exception as e:
            raise Exception(f"持久化失败: {str(e)}") from e
