"""
知识库管理器

本模块实现了知识库的管理功能，包括：
1. KnowledgeBase: 单个知识库的文档增删查操作
2. KnowledgeBaseManager: 多个知识库的创建、管理、删除

知识库的作用：
- 将文档按主题或业务分类管理
- 每个知识库独立维护自己的向量索引
- 支持多知识库并行检索

使用场景：
- "技术文档库"：存放技术文档
- "产品手册库"：存放产品说明
- "FAQ库"：存放常见问题
"""

import shutil
import json
from typing import List, Optional, Dict
from pathlib import Path
from datetime import datetime
from langchain_core.documents import Document

from src.core.vectorstore.base import BaseVectorStore
from src.core.embedding.base import BaseEmbedding


class KnowledgeBase:
    """
    单个知识库
    
    封装了向量存储和嵌入模型，提供文档的增删查操作。
    每个知识库都有独立的存储目录和向量索引。
    
    属性：
        name: 知识库名称（唯一标识）
        vectorstore: 向量存储实例（用于存储和检索文档向量）
        embedding: 嵌入模型实例（用于文本向量化）
        kb_path: 知识库在磁盘上的存储路径
        
    设计理念：
        - 单一职责：只负责单个知识库的操作
        - 封装性：隐藏向量存储的细节
        - 持久化：自动保存到磁盘
    """
    
    def __init__(
        self,
        name: str,
        vectorstore: BaseVectorStore,
        embedding: BaseEmbedding,
        kb_path: Optional[Path] = None
    ):
        """
        初始化知识库
        
        Args:
            name: 知识库名称，建议使用有意义的名称（如 "技术文档"）
            vectorstore: 向量存储实例（已初始化的 FAISSVectorStore 等）
            embedding: 嵌入模型实例（已初始化的 OpenAIEmbedding 等）
            kb_path: 知识库存储路径，默认为 ./data/knowledge_base/{name}
            
        注意：
            - kb_path 会自动创建，无需提前创建目录
            - vectorstore 需要配置正确的 persist_directory
        """
        self.name = name
        self.vectorstore = vectorstore
        self.embedding = embedding
        
        # 设置存储路径（如果未指定，使用默认路径）
        self.kb_path = kb_path or Path(f"./data/knowledge_base/{name}")
        
        # 创建目录（如果不存在）
        # parents=True: 创建所有必需的父目录
        # exist_ok=True: 如果目录已存在，不报错
        self.kb_path.mkdir(parents=True, exist_ok=True)
        
        # 文档索引文件路径
        self.doc_index_file = self.kb_path / "documents.json"
        
        # 加载或初始化文档索引
        self.documents_index = self._load_doc_index()
    
    def _load_doc_index(self) -> List[Dict]:
        """
        加载文档索引
        
        从 documents.json 文件中加载文档索引信息。
        如果文件不存在，返回空列表。
        
        Returns:
            List[Dict]: 文档索引列表，每个元素包含：
                - source: 文档来源（文件路径）
                - chunk_count: 文档块数量
                - added_at: 添加时间
                - updated_at: 最后更新时间
                
        索引示例：
            [
                {
                    "source": "docs/python.pdf",
                    "chunk_count": 15,
                    "added_at": "2024-01-01T10:00:00",
                    "updated_at": "2024-01-01T10:00:00"
                },
                ...
            ]
        """
        if self.doc_index_file.exists():
            try:
                with open(self.doc_index_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️ 加载文档索引失败: {e}")
                return []
        return []
    
    def _save_doc_index(self):
        """
        保存文档索引到磁盘
        
        将当前的文档索引保存到 documents.json 文件。
        使用 JSON 格式存储，便于人工查看和编辑。
        
        注意：
            - ensure_ascii=False: 支持中文字符
            - indent=2: 格式化输出，便于阅读
        """
        try:
            with open(self.doc_index_file, 'w', encoding='utf-8') as f:
                json.dump(self.documents_index, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️ 保存文档索引失败: {e}")
    
    def add_documents(self, documents: List[Document]) -> int:
        """
        添加文档到知识库
        
        工作流程：
            1. 检查文档列表是否为空
            2. 调用 vectorstore.add_documents() 添加文档
               - 这会自动对文档进行向量化（使用 embedding）
               - 将向量存储到向量索引中
            3. 调用 vectorstore.persist() 持久化到磁盘
               - 确保数据不会因程序退出而丢失
               
        Args:
            documents: 要添加的文档列表（Document 对象）
            
        Returns:
            int: 成功添加的文档数量
            
        Raises:
            Exception: 添加文档失败时抛出异常
            
        示例：
            >>> kb = KnowledgeBase("tech_docs", vectorstore, embedding)
            >>> docs = [Document(page_content="Python 教程", metadata={"source": "a.txt"})]
            >>> count = kb.add_documents(docs)
            >>> print(f"添加了 {count} 个文档")
        """
        try:
            # 检查文档列表是否为空
            if not documents:
                return 0
            
            # 添加文档到向量存储
            # 这里会自动：
            # 1. 对每个文档的 page_content 进行向量化
            # 2. 将向量和元数据存储到 FAISS 索引
            self.vectorstore.add_documents(documents)
            
            # 持久化到磁盘
            # 将内存中的向量索引保存到 kb_path
            self.vectorstore.persist()
            
            # 更新文档索引
            # 按来源分组统计文档块数量
            source_chunks = {}
            for doc in documents:
                source = doc.metadata.get('source', 'unknown')
                if source not in source_chunks:
                    source_chunks[source] = 0
                source_chunks[source] += 1
            
            # 更新或添加索引记录
            current_time = datetime.now().isoformat()
            for source, chunk_count in source_chunks.items():
                # 查找是否已存在该文档的索引
                existing = next(
                    (d for d in self.documents_index if d['source'] == source),
                    None
                )
                
                if existing:
                    # 已存在：更新块数量和时间
                    existing['chunk_count'] += chunk_count
                    existing['updated_at'] = current_time
                else:
                    # 不存在：添加新索引
                    self.documents_index.append({
                        'source': source,
                        'chunk_count': chunk_count,
                        'added_at': current_time,
                        'updated_at': current_time
                    })
            
            # 保存索引到磁盘
            self._save_doc_index()
            
            # 返回添加的文档数量
            return len(documents)
            
        except Exception as e:
            # 捕获所有异常，提供友好的错误信息
            raise Exception(f"添加文档到知识库 '{self.name}' 失败: {str(e)}") from e
    
    def search(self, query: str, k: int = 4) -> List[Document]:
        """
        在知识库中搜索相关文档
        
        工作流程：
            1. 将查询文本向量化（自动完成）
            2. 在向量索引中进行相似度搜索
            3. 返回最相关的 k 个文档
            
        Args:
            query: 查询文本（用户的问题）
            k: 返回的文档数量，默认 4
            
        Returns:
            List[Document]: 最相关的文档列表，按相似度降序排列
            
        Raises:
            Exception: 搜索失败时抛出异常
            
        注意：
            - 如果知识库为空，会返回空列表
            - 如果文档总数少于 k，返回所有文档
            
        示例：
            >>> results = kb.search("什么是 Python？", k=3)
            >>> for doc in results:
            ...     print(doc.page_content[:100])
        """
        try:
            # 检查向量库是否为空
            if self.vectorstore.vectorstore is None:
                return []
            
            # 调用向量存储的相似度搜索
            # similarity_search 会自动：
            # 1. 对 query 进行向量化
            # 2. 计算与所有文档向量的相似度（余弦相似度）
            # 3. 返回最相似的 k 个文档
            results = self.vectorstore.similarity_search(query, k=k)
            
            return results
            
        except Exception as e:
            raise Exception(f"在知识库 '{self.name}' 中搜索失败: {str(e)}") from e
    
    def delete(self):
        """
        删除知识库
        
        工作流程：
            1. 清空内存中的向量索引
            2. 删除磁盘上的存储目录（包括所有文件）
            
        注意：
            - 这是不可逆操作，请谨慎使用
            - 删除后，知识库对象仍然存在，但已无法使用
            - 建议在删除前提示用户确认
            
        Raises:
            Exception: 删除失败时抛出异常
            
        示例：
            >>> kb.delete()
            >>> # 知识库已被删除，无法再使用
        """
        try:
            # 步骤1: 清空向量存储（释放内存）
            # 这会将 vectorstore.vectorstore 设置为 None
            self.vectorstore.delete()
            
            # 步骤2: 删除磁盘上的存储目录
            if self.kb_path.exists():
                # shutil.rmtree() 递归删除整个目录树
                # 包括目录下的所有文件和子目录
                shutil.rmtree(self.kb_path)
                
        except Exception as e:
            raise Exception(f"删除知识库 '{self.name}' 失败: {str(e)}") from e
    
    def get_document_count(self) -> int:
        """
        获取知识库中的文档数量（不是chunk数量）
        
        Returns:
            int: 文档数量，如果知识库为空返回 0
            
        注意：
            返回的是实际文档文件数量，不是chunk数量
            一个文档可能被分成多个chunk
        """
        return len(self.documents_index)
    
    def upload_file(
        self,
        file_path: str,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        splitter_type: str = "recursive"
    ) -> int:
        """
        上传文件到知识库
        
        这是一个便捷方法，封装了完整的文档处理流程：
        加载 → 切分 → 向量化 → 存储
        
        工作流程：
            1. 使用 DocumentLoaderFactory 根据文件扩展名自动选择加载器
            2. 加载文件内容为 Document 对象
            3. 使用 TextSplitterFactory 切分文档为小块
            4. 调用 add_documents() 添加到向量库
            5. 自动更新文档索引
            
        Args:
            file_path: 文件路径（支持相对路径和绝对路径）
                支持的格式: .txt, .pdf, .md, .markdown
            chunk_size: 文本块大小，默认 500 字符
                建议范围: 200-1000
            chunk_overlap: 文本块重叠大小，默认 50 字符
                建议为 chunk_size 的 10%-20%
            splitter_type: 切分器类型，默认 "recursive"
                - "recursive": 递归切分器（通用，推荐）
                - "chinese": 中文切分器（针对中文优化）
                
        Returns:
            int: 成功添加的文档块数量
            
        Raises:
            Exception: 文件加载、切分或添加失败时抛出异常
            
        使用场景：
            - Web 界面文件上传
            - 批量导入文档目录
            - 命令行工具上传文件
            
        示例：
            >>> kb = KnowledgeBase("技术文档", vectorstore, embedding)
            >>> 
            >>> # 上传单个文件
            >>> count = kb.upload_file("docs/python_tutorial.pdf")
            >>> print(f"添加了 {count} 个文档块")
            >>> 
            >>> # 上传中文文档（使用中文切分器）
            >>> count = kb.upload_file(
            ...     "docs/chinese_doc.txt",
            ...     chunk_size=300,
            ...     splitter_type="chinese"
            ... )
        """
        from src.core.document import DocumentLoaderFactory, TextSplitterFactory
        
        try:
            # 保存原始文件名（用于显示）
            original_filename = Path(file_path).name
            
            # 步骤1: 加载文件
            # DocumentLoaderFactory 会根据文件扩展名自动选择加载器
            print(f"📄 正在加载文件: {original_filename}...")
            documents = DocumentLoaderFactory.load(file_path)
            
            if not documents:
                raise ValueError(f"文件 '{original_filename}' 加载后为空")
            
            # 更新文档的 metadata，使用原始文件名作为 source
            # 这样在文档列表中显示的就是原始文件名，而不是临时文件路径
            for doc in documents:
                # 保留原始文件名，而不是临时文件路径
                doc.metadata['source'] = original_filename
                doc.metadata['original_path'] = file_path  # 保留原始路径用于内部处理
            
            # 步骤2: 切分文档
            print(f"✂️ 正在切分文档: {original_filename}...")
            # TextSplitterFactory 根据类型选择切分策略
            chunks = TextSplitterFactory.split(
                documents=documents,
                splitter_type=splitter_type,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            
            if not chunks:
                raise ValueError(f"文件 '{original_filename}' 切分后为空")
            
            # 步骤3: 添加到向量库
            print(f"💾 正在添加文档到向量库: {original_filename} ({len(chunks)} 个分块)...")
            # add_documents() 会自动处理向量化、存储和索引更新
            count = self.add_documents(chunks)
            
            print(f"✅ 文件 '{original_filename}' 上传成功，共 {count} 个分块")
            return count
            
        except Exception as e:
            raise Exception(f"上传文件 '{file_path}' 到知识库 '{self.name}' 失败: {str(e)}") from e
    
    def upload_directory(
        self,
        directory_path: str,
        file_extensions: Optional[List[str]] = None,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        recursive: bool = True
    ) -> Dict[str, int]:
        """
        批量上传目录中的文件
        
        遍历目录，上传所有支持的文件到知识库。
        
        Args:
            directory_path: 目录路径
            file_extensions: 要处理的文件扩展名列表
                默认: ['.txt', '.pdf', '.md', '.markdown']
            chunk_size: 文本块大小
            chunk_overlap: 文本块重叠大小
            recursive: 是否递归处理子目录，默认 True
            
        Returns:
            Dict[str, int]: 处理结果字典
                - success: 成功处理的文件数
                - failed: 失败的文件数
                - total_chunks: 总文档块数
                - files: 每个文件的处理结果列表
                
        示例：
            >>> result = kb.upload_directory("./docs", recursive=True)
            >>> print(f"成功: {result['success']}, 失败: {result['failed']}")
            >>> print(f"总共添加了 {result['total_chunks']} 个文档块")
        """
        from pathlib import Path
        
        if file_extensions is None:
            file_extensions = ['.txt', '.pdf', '.md', '.markdown']
        
        directory = Path(directory_path)
        if not directory.exists():
            raise FileNotFoundError(f"目录不存在: {directory_path}")
        
        # 收集所有符合条件的文件
        pattern = "**/*" if recursive else "*"
        files_to_process = []
        
        for file_path in directory.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in file_extensions:
                files_to_process.append(file_path)
        
        # 处理每个文件
        result = {
            'success': 0,
            'failed': 0,
            'total_chunks': 0,
            'files': []
        }
        
        for file_path in files_to_process:
            try:
                count = self.upload_file(
                    str(file_path),
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )
                result['success'] += 1
                result['total_chunks'] += count
                result['files'].append({
                    'file': str(file_path),
                    'status': 'success',
                    'chunks': count
                })
            except Exception as e:
                result['failed'] += 1
                result['files'].append({
                    'file': str(file_path),
                    'status': 'failed',
                    'error': str(e)
                })
        
        return result
    
    def list_documents(self) -> List[Dict]:
        """
        列出知识库中的所有文档
        
        Returns:
            List[Dict]: 文档信息列表，每个元素包含：
                - source: 文档来源（文件路径）
                - chunk_count: 文档块数量
                - added_at: 添加时间
                - updated_at: 最后更新时间
                
        示例：
            >>> docs = kb.list_documents()
            >>> for doc in docs:
            ...     print(f"{doc['source']}: {doc['chunk_count']} 块")
        """
        return self.documents_index.copy()
    
    def get_document_info(self, source: str) -> Optional[Dict]:
        """
        获取指定文档的详细信息
        
        Args:
            source: 文档来源（文件路径）
            
        Returns:
            Optional[Dict]: 文档信息，如果不存在返回 None
        """
        return next(
            (d for d in self.documents_index if d['source'] == source),
            None
        )
    
    def delete_document(self, source: str):
        """
        删除指定来源的文档
        
        注意：
            由于 FAISS 不支持按 ID 删除单个文档，
            此方法会重建整个向量库（移除指定文档后重新添加其他文档）。
            
        工作流程：
            1. 从索引中查找要删除的文档
            2. 如果是唯一的文档，清空向量库
            3. 如果有其他文档，调用 rebuild_vectorstore 重建
            
        Args:
            source: 要删除的文档来源（文件路径）
            
        Raises:
            ValueError: 如果文档不存在
            Exception: 删除失败时抛出异常
            
        警告：
            - 此操作比较耗时，因为需要重建向量库
            - 如果知识库很大，建议谨慎使用
            
        示例：
            >>> kb.delete_document("docs/old_file.pdf")
        """
        try:
            # 查找文档
            doc_info = self.get_document_info(source)
            if not doc_info:
                raise ValueError(f"文档 '{source}' 不存在于知识库中")
            
            # 从索引中移除
            self.documents_index = [
                d for d in self.documents_index if d['source'] != source
            ]
            
            # 保存更新后的索引
            self._save_doc_index()
            
            # 如果索引为空，直接清空向量库
            if not self.documents_index:
                self.vectorstore.delete()
                print(f"✅ 文档 '{source}' 已删除，知识库已清空")
                return
            
            # 否则，重建向量库（不包含被删除的文档）
            print(f"⚠️ 正在重建向量库（移除 '{source}'）...")
            self.rebuild_vectorstore(exclude_sources=[source])
            print(f"✅ 文档 '{source}' 已删除")
            
        except ValueError:
            # 保留 ValueError，直接向上抛出
            raise
        except Exception as e:
            raise Exception(f"删除文档 '{source}' 失败: {str(e)}") from e
    
    def rebuild_vectorstore(
        self,
        new_embedding: Optional[BaseEmbedding] = None,
        exclude_sources: Optional[List[str]] = None
    ):
        """
        重建向量库
        
        使用场景：
            1. 更换 Embedding 模型后重新向量化所有文档
            2. 删除文档后重建向量库
            3. 向量库损坏时恢复
            4. 优化向量库性能
            
        工作流程：
            1. 保存当前文档索引信息
            2. 清空向量库
            3. 如果提供新 Embedding，更新模型
            4. 重新加载所有文档（排除指定的文档）
            5. 重新切分和向量化
            6. 添加到新向量库
            
        Args:
            new_embedding: 新的 Embedding 模型（可选）
                如果提供，将使用新模型重新向量化所有文档
            exclude_sources: 要排除的文档来源列表（可选）
                重建时不包含这些文档
                
        Raises:
            Exception: 重建失败时抛出异常
            
        注意：
            - 此操作比较耗时，尤其是文档量大时
            - 重建期间知识库暂时不可用
            - 需要确保原始文件仍然存在
            
        示例：
            >>> # 场景1: 更换 Embedding 模型
            >>> new_embedding = OpenAIEmbedding(model_name="text-embedding-3-small")
            >>> kb.rebuild_vectorstore(new_embedding=new_embedding)
            >>> 
            >>> # 场景2: 删除文档后重建
            >>> kb.rebuild_vectorstore(exclude_sources=["old_file.pdf"])
        """
        from src.core.document import DocumentLoaderFactory, TextSplitterFactory
        
        try:
            print(f"🔄 开始重建向量库 '{self.name}'...")
            
            # 保存原始文档列表
            original_docs = self.documents_index.copy()
            
            if not original_docs:
                print("⚠️ 知识库为空，无需重建")
                return
            
            # 过滤要排除的文档
            if exclude_sources:
                docs_to_rebuild = [
                    d for d in original_docs 
                    if d['source'] not in exclude_sources
                ]
            else:
                docs_to_rebuild = original_docs
            
            if not docs_to_rebuild:
                print("⚠️ 所有文档都被排除，清空向量库")
                self.vectorstore.delete()
                self.documents_index = []
                self._save_doc_index()
                return
            
            # 清空向量库
            self.vectorstore.delete()
            
            # 如果提供了新 Embedding，更新
            if new_embedding:
                print("📝 使用新的 Embedding 模型")
                self.embedding = new_embedding
                # 注意：这里可能需要重新创建 vectorstore
                # 具体取决于你的 VectorStore 实现
            
            # 清空文档索引（准备重新添加）
            self.documents_index = []
            
            # 重新加载并添加每个文档
            total_chunks = 0
            failed_files = []
            
            for doc_info in docs_to_rebuild:
                source = doc_info['source']
                try:
                    print(f"  处理: {source}")
                    
                    # 加载文档
                    documents = DocumentLoaderFactory.load(source)
                    
                    # 切分文档
                    chunks = TextSplitterFactory.split(
                        documents=documents,
                        splitter_type="recursive",
                        chunk_size=500,
                        chunk_overlap=50
                    )
                    
                    # 添加到向量库
                    count = self.add_documents(chunks)
                    total_chunks += count
                    
                except Exception as e:
                    print(f"  ⚠️ 失败: {source} - {e}")
                    failed_files.append(source)
            
            print(f"✅ 重建完成:")
            print(f"   成功: {len(docs_to_rebuild) - len(failed_files)} 个文档")
            print(f"   失败: {len(failed_files)} 个文档")
            print(f"   总块数: {total_chunks}")
            
            if failed_files:
                print(f"   失败文件: {', '.join(failed_files)}")
            
        except Exception as e:
            raise Exception(f"重建向量库失败: {str(e)}") from e


class KnowledgeBaseManager:
    """
    知识库管理器
    
    管理多个知识库的创建、获取、列表、删除操作。
    使用字典存储所有知识库实例，提供统一的管理接口。
    
    属性：
        root_path: 所有知识库的根目录
        knowledge_bases: 存储所有知识库的字典 {name: KnowledgeBase}
        
    设计模式：
        - 单例模式：通常整个应用只需要一个管理器实例
        - 工厂模式：负责创建和管理 KnowledgeBase 实例
        
    使用场景：
        - Web 应用：管理所有用户的知识库
        - API 服务：提供知识库的 CRUD 接口
        - 命令行工具：批量管理知识库
    """
    
    def __init__(self, root_path: Path = Path("./data/knowledge_base")):
        """
        初始化知识库管理器
        
        Args:
            root_path: 知识库根目录，所有知识库都存储在此目录下
                      默认为 ./data/knowledge_base
                      
        目录结构：
            root_path/
            ├── kb1/        # 知识库1的目录
            ├── kb2/        # 知识库2的目录
            └── ...
            
        注意：
            - root_path 会自动创建
            - 每个知识库会在 root_path 下创建独立的子目录
        """
        self.root_path = Path(root_path)
        # 创建根目录（如果不存在）
        self.root_path.mkdir(parents=True, exist_ok=True)
        
        # 初始化知识库字典
        # key: 知识库名称（字符串）
        # value: KnowledgeBase 实例
        self.knowledge_bases: dict[str, KnowledgeBase] = {}
    
    def create_kb(
        self,
        name: str,
        vectorstore: BaseVectorStore,
        embedding: BaseEmbedding,
    ) -> KnowledgeBase:
        """
        创建新的知识库
        
        工作流程：
            1. 检查知识库名称是否已存在（避免重复）
            2. 为知识库创建独立的存储目录
            3. 创建 KnowledgeBase 实例
            4. 将实例添加到管理字典
            5. 返回创建的知识库
            
        Args:
            name: 知识库名称（唯一标识）
                 - 建议使用有意义的名称，如 "技术文档"、"产品手册"
                 - 名称不能包含特殊字符（/, \, :, * 等）
            vectorstore: 向量存储实例
                 - 通常是 FAISSVectorStore 的实例
                 - 需要配置正确的 persist_directory
            embedding: 嵌入模型实例
                 - 通常是 OpenAIEmbedding 或 OllamaEmbedding
                 - 同一个知识库应使用相同的 embedding 模型
                 
        Returns:
            KnowledgeBase: 创建的知识库实例
            
        Raises:
            ValueError: 如果知识库名称已存在
            Exception: 创建过程中发生错误
            
        示例：
            >>> manager = KnowledgeBaseManager()
            >>> vectorstore = FAISSVectorStore(persist_directory="./data/kb1")
            >>> embedding = OpenAIEmbedding(model_name="text-embedding-ada-002")
            >>> kb = manager.create_kb("技术文档", vectorstore, embedding)
            >>> print(f"创建了知识库: {kb.name}")
        """
        try:
            # 检查名称是否已存在
            # 注意：这里的逻辑要正确！
            if name in self.knowledge_bases:  # ✅ 如果已存在，报错
                raise ValueError(f"知识库 '{name}' 已存在，请使用不同的名称")
            
            # 为知识库创建独立的存储目录
            kb_path = self.root_path / name
            kb_path.mkdir(parents=True, exist_ok=True)
            
            # 创建 KnowledgeBase 实例
            kb = KnowledgeBase(
                name=name,
                vectorstore=vectorstore,
                embedding=embedding,
                kb_path=kb_path
            )
            
            # 添加到管理字典
            self.knowledge_bases[name] = kb
            
            return kb
            
        except ValueError as e:
            # 重新抛出 ValueError（名称重复）
            raise e
        except Exception as e:
            # 捕获其他异常，提供友好的错误信息
            raise Exception(f"创建知识库 '{name}' 失败: {str(e)}") from e
    
    def get_kb(self, name: str) -> Optional[KnowledgeBase]:
        """
        获取指定名称的知识库
        
        Args:
            name: 知识库名称
            
        Returns:
            Optional[KnowledgeBase]:
                - 如果找到，返回 KnowledgeBase 实例
                - 如果未找到，返回 None
                
        注意：
            - 不会抛出异常，未找到时返回 None
            - 调用者需要检查返回值是否为 None
            
        示例：
            >>> kb = manager.get_kb("技术文档")
            >>> if kb:
            ...     print(f"找到知识库: {kb.name}")
            ... else:
            ...     print("知识库不存在")
        """
        # dict.get() 方法：
        # - 如果 key 存在，返回对应的 value
        # - 如果 key 不存在，返回 None（默认值）
        return self.knowledge_bases.get(name)
    
    def list_kb(self) -> List[str]:
        """
        列出所有知识库的名称
        
        Returns:
            List[str]: 知识库名称列表
            
        注意：
            - 返回的是名称列表，不是 KnowledgeBase 实例
            - 如果没有知识库，返回空列表 []
            
        示例：
            >>> kb_names = manager.list_kb()
            >>> print(f"共有 {len(kb_names)} 个知识库")
            >>> for name in kb_names:
            ...     print(f"  - {name}")
        """
        # dict.keys() 返回字典的所有 key
        # list() 将其转换为列表
        return list(self.knowledge_bases.keys())
    
    def delete_kb(self, name: str):
        """
        删除指定的知识库
        
        工作流程：
            1. 检查知识库是否存在
            2. 调用知识库的 delete() 方法（清空向量索引、删除文件）
            3. 从管理字典中移除
            
        Args:
            name: 要删除的知识库名称
            
        Raises:
            ValueError: 如果知识库不存在
            Exception: 删除过程中发生错误
            
        注意：
            - 这是不可逆操作，删除后无法恢复
            - 会删除磁盘上的所有文件
            - 建议在调用前提示用户确认
            
        示例：
            >>> if manager.get_kb("旧知识库"):
            ...     manager.delete_kb("旧知识库")
            ...     print("知识库已删除")
        """
        try:
            # 从字典中获取知识库
            kb = self.knowledge_bases.get(name)
            
            # 检查知识库是否存在
            if not kb:
                raise ValueError(f"知识库 '{name}' 不存在，无法删除")
            
            # 调用知识库的 delete() 方法
            # 这会清空向量索引并删除磁盘文件
            kb.delete()
            
            # 从管理字典中移除
            del self.knowledge_bases[name]
            
        except ValueError as e:
            # 重新抛出 ValueError（知识库不存在）
            raise e
        except Exception as e:
            # 捕获其他异常
            raise Exception(f"删除知识库 '{name}' 失败: {str(e)}") from e
    
    def get_kb_info(self, name: str) -> dict:
        """
        获取知识库的详细信息
        
        Args:
            name: 知识库名称
            
        Returns:
            dict: 知识库信息，包含：
                - name: 名称
                - path: 存储路径
                - document_count: 文档数量
                - exists: 是否存在
                
        这是一个辅助方法，用于显示和调试。
        """
        kb = self.get_kb(name)
        
        if not kb:
            return {
                "name": name,
                "exists": False
            }
        
        return {
            "name": kb.name,
            "path": str(kb.kb_path),
            "document_count": kb.get_document_count(),
            "exists": True
        }

