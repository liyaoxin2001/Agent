"""
文本切分器模块

本模块实现了多种文本切分策略，用于将长文档切分成适合向量化的小块（Chunks）。

为什么需要文本切分？
    1. 提高检索精度：小块文本更容易精确匹配用户查询
    2. 控制 Token 消耗：LLM 有输入长度限制（如 GPT-3.5 的 4K tokens）
    3. 提升相关性：避免无关信息干扰答案生成
    4. 优化向量化：向量模型对短文本的表示更准确

核心概念：
    - chunk_size: 每个文本块的最大字符数
    - chunk_overlap: 相邻文本块之间的重叠字符数
    
    重叠的作用：
        - 避免语义断裂：防止完整的句子或段落被切断
        - 保持上下文连贯性：相邻块之间保持一定的语义联系
        
示例：
    原文: "人工智能是计算机科学的一个分支。它专注于创建智能机器。"
    chunk_size=20, chunk_overlap=5
    
    结果:
    Chunk 1: "人工智能是计算机科学的"     (20 字符)
    Chunk 2:     "科学的一个分支。它专注"  (overlap: "科学的")
    Chunk 3:         "。它专注于创建智能机器" (overlap: "。它专注")
"""

from abc import ABC, abstractmethod
from typing import List, Optional
import re
from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter as LCRecursiveTextSplitter,
    CharacterTextSplitter,
)


class BaseTextSplitter(ABC):
    """
    文本切分器抽象基类
    
    定义了所有文本切分器必须实现的接口。
    
    设计模式：模板方法模式
        - 定义算法骨架（split_documents）
        - 子类实现具体切分策略
        - 统一的接口便于替换不同的切分器
        
    核心参数：
        - chunk_size: 块大小（字符数）
        - chunk_overlap: 重叠大小（字符数）
    """
    
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        length_function: Optional[callable] = None
    ):
        """
        初始化文本切分器
        
        Args:
            chunk_size: 每个文本块的最大字符数，默认 500
                - 过小：切分过细，可能丢失上下文
                - 过大：切分过粗，检索不精确
                - 建议：200-1000，根据文档类型调整
                
            chunk_overlap: 相邻文本块的重叠字符数，默认 50
                - 建议：chunk_size 的 10%-20%
                - 示例：chunk_size=500, overlap=50-100
                
            length_function: 自定义长度计算函数，默认 len()
                - 可以自定义为 token 计数函数
                - 示例：tiktoken.encoding_for_model("gpt-3.5-turbo").encode
        
        参数选择建议：
            文档类型          chunk_size  chunk_overlap
            技术文档           300-500       50-100
            新闻文章           500-800      100-150
            学术论文           800-1200     150-200
            对话记录           200-300       30-50
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.length_function = length_function or len
        
        # 验证参数
        if chunk_size <= 0:
            raise ValueError("chunk_size 必须大于 0")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap 不能为负数")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap 必须小于 chunk_size")
    
    @abstractmethod
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        切分文档列表
        
        Args:
            documents: 待切分的文档列表
            
        Returns:
            List[Document]: 切分后的文档块列表
            
        注意：
            - 每个切分后的块仍然是 Document 对象
            - metadata 会被继承到每个切分块
            - 可以在 metadata 中添加块的位置信息
        """
        pass
    
    def split_text(self, text: str) -> List[str]:
        """
        切分纯文本（不包含 metadata）
        
        Args:
            text: 待切分的文本字符串
            
        Returns:
            List[str]: 切分后的文本块列表
            
        这是一个辅助方法，便于直接处理字符串。
        """
        # 将文本转为 Document，然后切分，最后提取文本
        doc = Document(page_content=text)
        split_docs = self.split_documents([doc])
        return [d.page_content for d in split_docs]


class RecursiveTextSplitter(BaseTextSplitter):
    """
    递归字符文本切分器
    
    这是最常用、最智能的文本切分器。
    
    工作原理：
        1. 首先尝试按段落分割（\\n\\n）
        2. 如果段落仍太长，按句子分割（。！？）
        3. 如果句子仍太长，按短句分割（，、；）
        4. 最后按字符分割
        
    优势：
        - 智能分割：优先保持语义完整性
        - 自适应：根据文本结构自动调整策略
        - 通用性强：适用于大多数文本类型
        
    使用场景：
        - 通用文档切分（推荐）
        - 技术文档
        - 新闻文章
        - 学术论文
        
    示例：
        >>> splitter = RecursiveTextSplitter(
        ...     chunk_size=500,
        ...     chunk_overlap=50
        ... )
        >>> docs = splitter.split_documents(documents)
    """
    
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separators: Optional[List[str]] = None,
        keep_separator: bool = True
    ):
        """
        初始化递归文本切分器
        
        Args:
            chunk_size: 块大小
            chunk_overlap: 重叠大小
            separators: 分隔符列表（按优先级排序）
                默认: ["\n\n", "\n", "。", "！", "？", ".", "!", "?", ";", "；", ",", "，", " ", ""]
            keep_separator: 是否保留分隔符，默认 True
                - True: "你好。世界" -> ["你好。", "世界"]
                - False: "你好。世界" -> ["你好", "世界"]
        """
        super().__init__(chunk_size, chunk_overlap)
        
        # 默认分隔符（按优先级从高到低）
        if separators is None:
            separators = [
                "\n\n",   # 段落分隔
                "\n",     # 行分隔
                "。",     # 中文句号
                "！",     # 中文感叹号
                "？",     # 中文问号
                ".",      # 英文句号
                "!",      # 英文感叹号
                "?",      # 英文问号
                ";",      # 英文分号
                "；",     # 中文分号
                ",",      # 英文逗号
                "，",     # 中文逗号
                " ",      # 空格
                "",       # 字符级分割（最后的兜底策略）
            ]
        
        self.separators = separators
        self.keep_separator = keep_separator
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        切分文档列表
        
        工作流程：
            1. 遍历每个 Document
            2. 提取 page_content 和 metadata
            3. 使用 LangChain 的 RecursiveCharacterTextSplitter 切分
            4. 将切分结果转为 Document 对象
            5. 继承原始 metadata
            
        Args:
            documents: 待切分的文档列表
            
        Returns:
            List[Document]: 切分后的文档块列表
            
        示例：
            输入: [Document(page_content="很长的文本...", metadata={"source": "a.txt"})]
            输出: [
                Document(page_content="第一块...", metadata={"source": "a.txt"}),
                Document(page_content="第二块...", metadata={"source": "a.txt"}),
                ...
            ]
        """
        try:
            # 创建 LangChain 的递归切分器
            splitter = LCRecursiveTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=self.length_function,
                separators=self.separators,
                keep_separator=self.keep_separator,
            )
            
            # 使用 LangChain 的 split_documents 方法
            # 这个方法会自动处理 metadata 的继承
            split_docs = splitter.split_documents(documents)
            
            return split_docs
            
        except Exception as e:
            raise Exception(f"文档切分失败: {str(e)}") from e


class ChineseTextSplitter(BaseTextSplitter):
    """
    中文文本切分器
    
    专门针对中文文本优化的切分器。
    
    中文特点：
        - 没有空格分隔词语
        - 句子结构与英文不同
        - 标点符号使用习惯不同
        
    优化策略：
        - 优先按段落切分
        - 保持句子完整性
        - 考虑中文标点符号
        - 智能处理中英混合文本
        
    使用场景：
        - 中文技术文档
        - 中文新闻
        - 中文学术论文
        - 中英混合文本
        
    示例：
        >>> splitter = ChineseTextSplitter(
        ...     chunk_size=300,
        ...     chunk_overlap=30
        ... )
        >>> docs = splitter.split_documents(chinese_documents)
    """
    
    def __init__(
        self,
        chunk_size: int = 300,
        chunk_overlap: int = 30,
        pdf_mode: bool = False
    ):
        """
        初始化中文文本切分器
        
        Args:
            chunk_size: 块大小，中文建议 200-500
            chunk_overlap: 重叠大小，建议 chunk_size 的 10%
            pdf_mode: PDF 模式，是否针对 PDF 提取的文本优化
                - True: 处理 PDF 特有的格式问题（如换行、空格）
                - False: 普通文本模式
        """
        super().__init__(chunk_size, chunk_overlap)
        self.pdf_mode = pdf_mode
        
        # 中文句子分隔符（按优先级）
        self.chinese_separators = [
            "\n\n",    # 段落
            "\n",      # 行
            "。",      # 句号
            "！",      # 感叹号
            "？",      # 问号
            "；",      # 分号
            "，",      # 逗号
        ]
    
    def _preprocess_text(self, text: str) -> str:
        """
        预处理文本（针对中文优化）
        
        Args:
            text: 原始文本
            
        Returns:
            str: 预处理后的文本
            
        处理内容：
            1. 移除多余的空白字符
            2. 统一换行符
            3. 处理 PDF 提取的格式问题
            4. 保留必要的标点符号
        """
        # 统一换行符
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        if self.pdf_mode:
            # PDF 模式：处理常见的 PDF 提取问题
            # 1. 移除行内多余空格（PDF 提取常见问题）
            text = re.sub(r' +', ' ', text)
            # 2. 移除中文字符间的空格
            text = re.sub(r'([\u4e00-\u9fa5]) +([\u4e00-\u9fa5])', r'\1\2', text)
        
        # 移除多余的空行（超过 2 个连续换行）
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text.strip()
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        切分中文文档
        
        工作流程：
            1. 预处理每个文档的文本
            2. 使用递归策略切分
            3. 保持元数据不变
            
        Args:
            documents: 待切分的文档列表
            
        Returns:
            List[Document]: 切分后的文档块列表
        """
        try:
            # 预处理文档
            processed_docs = []
            for doc in documents:
                # 预处理文本内容
                processed_content = self._preprocess_text(doc.page_content)
                # 创建新的 Document（保留原始 metadata）
                processed_doc = Document(
                    page_content=processed_content,
                    metadata=doc.metadata.copy()
                )
                processed_docs.append(processed_doc)
            
            # 创建递归切分器（使用中文分隔符）
            splitter = LCRecursiveTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=self.length_function,
                separators=self.chinese_separators,
                keep_separator=True,
            )
            
            # 切分文档
            split_docs = splitter.split_documents(processed_docs)
            
            return split_docs
            
        except Exception as e:
            raise Exception(f"中文文档切分失败: {str(e)}") from e
    
    def split_text_by_sentences(self, text: str) -> List[str]:
        """
        按句子切分中文文本（辅助方法）
        
        Args:
            text: 中文文本
            
        Returns:
            List[str]: 句子列表
            
        这个方法用于将文本严格按句子分割，不考虑 chunk_size。
        适用于需要保持句子完整性的场景。
        
        示例：
            >>> splitter = ChineseTextSplitter()
            >>> text = "人工智能很重要。它改变了世界。我们要学习它。"
            >>> sentences = splitter.split_text_by_sentences(text)
            >>> print(sentences)
            ['人工智能很重要。', '它改变了世界。', '我们要学习它。']
        """
        # 使用正则表达式按中文标点切分
        # 保留标点符号
        sentences = re.split(r'([。！？\n])', text)
        
        # 重新组合句子和标点
        result = []
        for i in range(0, len(sentences) - 1, 2):
            sentence = sentences[i] + (sentences[i + 1] if i + 1 < len(sentences) else '')
            sentence = sentence.strip()
            if sentence:
                result.append(sentence)
        
        return result


class TextSplitterFactory:
    """
    文本切分器工厂类
    
    根据文档类型或语言自动选择合适的切分器。
    
    设计模式：工厂模式 + 策略模式
        - 封装切分器的创建逻辑
        - 根据条件选择最优策略
        
    使用示例：
        >>> # 自动选择切分器
        >>> splitter = TextSplitterFactory.get_splitter("chinese")
        >>> 
        >>> # 或者使用便捷方法
        >>> chunks = TextSplitterFactory.split(documents, "chinese")
    """
    
    @classmethod
    def get_splitter(
        cls,
        splitter_type: str = "recursive",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        **kwargs
    ) -> BaseTextSplitter:
        """
        获取指定类型的切分器
        
        Args:
            splitter_type: 切分器类型
                - "recursive": 递归切分器（默认，通用）
                - "chinese": 中文切分器
            chunk_size: 块大小
            chunk_overlap: 重叠大小
            **kwargs: 其他参数传递给切分器
            
        Returns:
            BaseTextSplitter: 对应的切分器实例
            
        示例：
            >>> # 通用切分器
            >>> splitter = TextSplitterFactory.get_splitter("recursive")
            >>> 
            >>> # 中文切分器（PDF 模式）
            >>> splitter = TextSplitterFactory.get_splitter(
            ...     "chinese",
            ...     chunk_size=300,
            ...     pdf_mode=True
            ... )
        """
        if splitter_type == "recursive":
            return RecursiveTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                **kwargs
            )
        elif splitter_type == "chinese":
            return ChineseTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                **kwargs
            )
        else:
            raise ValueError(
                f"不支持的切分器类型: {splitter_type}\n"
                f"支持的类型: recursive, chinese"
            )
    
    @classmethod
    def split(
        cls,
        documents: List[Document],
        splitter_type: str = "recursive",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        **kwargs
    ) -> List[Document]:
        """
        便捷方法：直接切分文档
        
        Args:
            documents: 待切分的文档列表
            splitter_type: 切分器类型
            chunk_size: 块大小
            chunk_overlap: 重叠大小
            **kwargs: 其他参数
            
        Returns:
            List[Document]: 切分后的文档块列表
            
        示例：
            >>> # 最简单的使用方式
            >>> chunks = TextSplitterFactory.split(
            ...     documents,
            ...     splitter_type="chinese",
            ...     chunk_size=300
            ... )
        """
        splitter = cls.get_splitter(
            splitter_type=splitter_type,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            **kwargs
        )
        return splitter.split_documents(documents)


