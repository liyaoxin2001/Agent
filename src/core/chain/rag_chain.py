"""
RAG Chain 实现

这是 RAG 系统的核心组件，负责协调 LLM 和 VectorStore 完成检索增强生成。

核心功能：
1. 从向量库检索相关文档
2. 组装包含上下文的 Prompt
3. 调用 LLM 生成答案
4. 支持流式生成
"""
from typing import List, Optional
from langchain_core.documents import Document

# 导入自定义模块
from src.core.llm.base import BaseLLM
from src.core.vectorstore.base import BaseVectorStore


class RAGChain:
    """
    RAG (Retrieval-Augmented Generation) 链
    
    RAG 是一种结合检索和生成的技术，通过从知识库中检索相关信息，
    然后将这些信息作为上下文提供给 LLM，生成更准确、更可靠的答案。
    
    核心工作流程：
    1. 接收用户问题
    2. 从向量数据库检索相关文档（基于语义相似度）
    3. 将检索到的文档组装成结构化的上下文
    4. 使用 Prompt 模板组合上下文和问题
    5. 调用 LLM 生成答案
    6. 返回生成的答案
    
    主要特性：
    - 支持自定义 Prompt 模板
    - 提供企业级默认模板
    - 支持流式和非流式生成
    - 完善的异常处理
    
    使用场景：
    - 企业知识库问答
    - 文档检索与问答
    - 智能客服系统
    - 技术文档助手
    """
    
    def __init__(
        self,
        llm: BaseLLM,
        vectorstore: BaseVectorStore,
        prompt_template: Optional[str] = None
    ):
        """
        初始化 RAG Chain
        
        Args:
            llm: LLM 实例（OpenAILLM 或 OllamaLLM）
            vectorstore: 向量存储实例（FAISSVectorStore）
            prompt_template: 自定义 Prompt 模板（可选）
                           如果不提供，将使用企业级默认模板
                           模板必须包含 {context} 和 {question} 占位符
        
        工作流程：
        1. 保存 LLM 和 VectorStore 的引用
        2. 设置 Prompt 模板：
           - 如果用户提供了 prompt_template，使用用户的
           - 如果用户没提供（None），调用 _get_default_template() 获取企业级模板
        """
        self.llm = llm
        self.vectorstore = vectorstore
        
        # 设置 Prompt 模板
        # prompt_template or self._get_default_template()
        # 逻辑：如果 prompt_template 为真（用户提供），使用它；否则调用默认方法
        self.prompt_template = prompt_template or self._get_default_template()
    
    def _get_default_template(self) -> str:
        """
        获取默认 prompt 模板（企业级）
        
        Returns:
            str: 包含 {context} 和 {question} 占位符的专业 Prompt 模板
        """
        template = """你是一位专业的智能助手，擅长基于提供的上下文信息进行精准、客观的回答。

## 上下文信息
{context}

## 用户问题
{question}

## 回答指南
请严格遵循以下要求提供高质量的回答：

1. **准确性优先**：仅基于上述上下文信息回答，确保信息准确可靠
2. **完整性保证**：如果上下文中包含相关信息，请提供详细、全面的回答
3. **诚实原则**：如果上下文中没有足够信息回答问题，请明确说明"根据提供的信息，我无法完整回答这个问题"
4. **禁止臆测**：严禁编造、推测或添加上下文中不存在的信息
5. **结构化表达**：使用清晰的逻辑结构组织回答，必要时使用要点列举
6. **简洁专业**：语言简洁明了，避免冗余，保持专业性

## 你的回答
"""
        return template
    
    def query(self, question: str, k: int = 4) -> str:
        """
        执行 RAG 查询（核心方法）
        
        这是 RAG Chain 的核心方法，实现了完整的检索增强生成流程。
        
        工作流程详解：
        1. 向量检索：使用向量数据库检索与问题最相关的 k 个文档
        2. 边界检查：处理未找到文档的情况，返回友好提示
        3. 上下文组装：将检索到的文档格式化为结构化上下文
           - 为每个文档添加编号（便于追溯来源）
           - 使用双换行分隔文档（增强可读性）
        4. Prompt 组装：使用模板将上下文和问题组合成完整 Prompt
        5. LLM 生成：调用语言模型生成答案
        6. 返回答案：返回生成的文本结果
        
        Args:
            question: 用户的问题文本
                     会被转换为向量进行相似度搜索
            k: 检索的文档数量，默认 4 个
               建议范围：3-10
               - 太少：可能遗漏重要信息
               - 太多：引入噪声，增加 token 消耗，降低准确性
        
        Returns:
            str: LLM 生成的答案文本
                 基于检索到的上下文信息回答用户问题
        
        Raises:
            Exception: 当 RAG 查询过程中发生错误时抛出
                      包装了底层异常，提供更友好的错误信息
        
        示例：
            >>> rag_chain = RAGChain(llm=my_llm, vectorstore=my_vectorstore)
            >>> answer = rag_chain.query("什么是 Python？")
            >>> print(answer)
            "Python 是一种高级编程语言..."
        
        性能考虑：
        - 向量检索通常是最耗时的步骤（取决于向量库大小）
        - LLM 生成时间取决于答案长度和模型选择
        - 总耗时 = 检索时间 + 生成时间（通常 1-5 秒）
        """
        try:
            # ===== 步骤 1: 向量检索 =====
            # 使用 similarity_search 从向量库中检索最相关的文档
            # 原理：将问题转换为向量，计算与文档向量的相似度
            relevant_docs = self.vectorstore.similarity_search(question, k=k)
            
            # ===== 步骤 2: 边界检查 =====
            # 处理边界情况：未检索到任何文档
            # 可能原因：向量库为空、问题与文档完全不相关
            if not relevant_docs:
                return "抱歉，在知识库中没有找到相关信息。"
            
            # ===== 步骤 3: 组装上下文 =====
            # 将检索到的文档格式化为结构化的上下文字符串
            contexts = []
            for i, doc in enumerate(relevant_docs, 1):
                # 为每个文档添加编号，格式：[文档1]\n文档内容
                # 编号的作用：
                # 1. 便于追溯信息来源
                # 2. 帮助 LLM 理解文档结构
                # 3. 提高回答的可信度（可以引用文档编号）
                context_piece = f"[文档{i}]\n{doc.page_content}"
                contexts.append(context_piece)
            
            # 使用双换行连接所有文档
            # 双换行的作用：在文档之间创建明显的视觉分隔
            context = "\n\n".join(contexts)

            # ===== 步骤 4: 组装 Prompt =====
            # 使用 Prompt 模板将上下文和问题组合成完整的 Prompt
            # format() 方法会替换模板中的 {context} 和 {question} 占位符
            prompt = self.prompt_template.format(context=context, question=question)

            # ===== 步骤 5: LLM 生成答案 =====
            # 调用 LLM 的 generate 方法生成答案
            # LLM 会基于 Prompt 中的上下文信息回答问题
            answer = self.llm.generate(prompt)

            # ===== 步骤 6: 返回答案 =====
            return answer
            
        except Exception as e:
            # 捕获并重新抛出异常，提供更友好的错误信息
            # from e: 保留原始异常的堆栈信息，便于调试
            raise Exception(f"RAG查询失败：{str(e)}") from e


    
    def stream_query(self, question: str, k: int = 4):
        """
        流式执行 RAG 查询
        
        与 query() 的核心区别在于答案的返回方式：
        - query(): 等待完整答案生成后一次性返回
        - stream_query(): 逐个产出文本片段（边生成边返回）
        
        流式生成的优势：
        1. 更好的用户体验：用户无需等待，实时看到生成过程
        2. 更低的首字延迟：第一个字的延迟大幅降低
        3. 适合长文本：生成长答案时体验更佳
        4. 可实现打字机效果：在 UI 中逐字显示
        
        应用场景：
        - Web 应用的实时对话
        - 聊天机器人界面
        - 需要流式输出的场景
        
        Args:
            question: 用户的问题文本
            k: 检索的文档数量，默认 4 个
        
        Yields:
            str: 生成的文本片段
                 每次 yield 返回一小段文本（通常是几个字符到几个单词）
                 调用者需要收集这些片段组成完整答案
        
        Raises:
            Exception: 当流式 RAG 查询失败时抛出
        
        示例：
            >>> # 打字机效果
            >>> print("问题: 什么是 Python？")
            >>> print("答案: ", end="", flush=True)
            >>> for chunk in rag_chain.stream_query("什么是 Python？"):
            ...     print(chunk, end="", flush=True)
            >>> print()  # 换行
        
        工作流程：
        1. 检索相关文档（与 query() 相同）
        2. 组装上下文（与 query() 相同）
        3. 组装 Prompt（与 query() 相同）
        4. 流式生成：使用 stream_generate() 而不是 generate()
        5. 逐个产出：使用 yield 逐个返回文本片段
        
        技术细节：
        - 使用 Python 生成器（Generator）实现
        - yield 关键字产出值后暂停，等待下次迭代
        - 不会一次性占用大量内存
        """
        try:
            # ===== 步骤 1-3: 检索和组装（与 query() 相同） =====
            
            # 向量检索
            relevant_docs = self.vectorstore.similarity_search(question, k=k)
            
            # 边界检查
            if not relevant_docs:
                yield "抱歉，我在知识库中没有找到相关信息。"
                return  # 提前结束生成器

            # 组装上下文（使用列表推导式，更简洁）
            contexts = [
                f"[文档{i}]\n{doc.page_content}"
                for i, doc in enumerate(relevant_docs, 1)
            ]
            context = "\n\n".join(contexts)
            
            # 组装 Prompt
            prompt = self.prompt_template.format(context=context, question=question)

            # ===== 步骤 4: 流式生成 =====
            # 使用 stream_generate() 而不是 generate()
            # stream_generate() 返回一个生成器，逐个产出文本片段
            for chunk in self.llm.stream_generate(prompt):
                # yield 产出每个文本片段
                # 调用者可以立即获取并处理这个片段
                yield chunk

        except Exception as e:
            # 异常处理：捕获并重新抛出
            raise Exception(f"流式RAG查询失败：{str(e)}") from e


# TODO: 实现 ConversationalRAGChain 类
# 提示：
# 1. 继承 RAGChain 或重新实现
# 2. 添加对话历史管理
# 3. 使用 LangChain 的 ConversationBufferMemory
# 4. 在 prompt 中包含历史对话
# 参考: https://python.langchain.com/docs/modules/memory/

class ConversationalRAGChain(RAGChain):
    """
    带对话历史的 RAG Chain（多轮对话）
    
    在基础 RAGChain 的基础上增加了对话历史管理功能，支持多轮对话。
    
    核心特性：
    1. 自动管理对话历史（问题-答案对）
    2. 在生成答案时考虑历史上下文
    3. 支持指代消解（如"它"、"他"等代词的理解）
    4. 自动限制历史长度，避免 token 超限
    
    与父类 RAGChain 的区别：
    - 父类：无状态，每次查询独立
    - 子类：有状态，维护对话历史
    
    使用场景：
    - 多轮对话问答系统
    - 需要上下文理解的聊天机器人
    - 连续追问的技术支持
    - 需要指代消解的对话场景
    
    工作原理：
    1. 存储每轮对话的（问题, 答案）对
    2. 查询时将历史格式化后加入 Prompt
    3. LLM 基于历史理解当前问题
    4. 生成答案后自动保存到历史
    
    示例场景：
        用户: "Python 是什么？"
        助手: "Python 是一种高级编程语言。"
        
        用户: "它有什么特点？"  # "它" 指代 Python
        助手: "Python 的特点包括..."  # 能理解 "它" = Python
    """
    
    def __init__(
        self,
        llm: BaseLLM,
        vectorstore: BaseVectorStore,
        prompt_template: Optional[str] = None,
        max_history: int = 100
    ):
        """
        初始化对话式 RAG Chain
        
        Args:
            llm: LLM 实例
            vectorstore: 向量存储实例
            prompt_template: 自定义 Prompt 模板（可选）
                           如果不提供，使用带历史的默认模板
            max_history: 最大保留的历史对话轮数，默认 100
                        超过此数量会自动删除最早的对话
                        设置原则：
                        - 太小：丢失重要上下文
                        - 太大：可能超过 LLM 的 token 限制
                        - 建议：5-20 轮（取决于对话复杂度）
        
        初始化流程：
        1. 调用父类 __init__，初始化基础 RAG 功能
        2. 创建空的对话历史列表
        3. 设置历史长度限制
        """
        # 调用父类初始化
        # super() 返回父类，调用其 __init__ 方法
        super().__init__(llm, vectorstore, prompt_template)
        
        # 对话历史存储
        # 数据结构：[(question1, answer1), (question2, answer2), ...]
        # 列表保持插入顺序，最新的对话在末尾
        self.chat_history = []
        
        # 历史长度限制
        # 防止历史过长导致 token 超限
        self.max_history = max_history

    def _get_default_template(self) -> str:
        """
        获取带历史的默认 Prompt 模板
        
        与父类模板的区别：
        - 父类：只包含 {context} 和 {question}
        - 子类：额外包含 {history} 占位符
        
        模板设计要点：
        1. 明确告诉 LLM 有历史信息
        2. 历史放在上下文之前（时间顺序）
        3. 指示 LLM 如何使用历史
        
        Returns:
            str: 包含 {history}, {context}, {question} 的模板字符串
        """
        template = """你是一个智能助手。基于以下对话历史和上下文信息回答用户的问题。

        对话历史：
        {history}

        上下文信息：
        {context}

        当前问题：
        {question}

        请基于对话历史和上下文信息回答。如果问题与之前的对话相关，请考虑历史信息。

        你的回答："""
        return template

    def query(self, question: str, k: int = 4) -> str:
        """
        带历史的 RAG 查询
        
        与父类的区别：
        - 包含对话历史
        - 保存当前对话到历史
        """
        try:
            # 检索文档
            docs = self.vectorstore.similarity_search(question, k=k)
            if not docs:
                answer = "抱歉，我在知识库中没有找到相关信息。"
                self._add_to_history(question, answer)
                return answer

            # 组装上下文
            context = "\n\n".join([
                f"[文档{i}]\n{doc.page_content}"
                for i, doc in enumerate(docs, 1)
            ])

            # 组装历史
            history = self._format_history()

            # 组装 Prompt
            prompt = self.prompt_template.format(
                history=history,
                context=context,
                question=question
            )
            
            # LLM 生成答案
            answer = self.llm.generate(prompt)

            # 保存到历史
            self._add_to_history(question, answer)
            return answer
            
        except Exception as e:
            raise Exception(f"对话 RAG 查询失败: {str(e)}") from e

    def _format_history(self) -> str:
        """
        格式化对话历史为字符串
        
        将内部存储的对话历史列表转换为 Prompt 可用的字符串格式。
        
        格式化规则：
        - 第一轮对话：返回特殊提示
        - 后续对话：格式化为 "用户: ... \n 助手: ..." 的形式
        
        Returns:
            str: 格式化后的历史字符串
                 如果是第一轮对话，返回 "（这是第一轮对话）"
                 否则返回所有历史对话的格式化文本
        
        示例输出：
            用户: Python 是什么？
            助手: Python 是一种高级编程语言。
            用户: 它有什么特点？
            助手: Python 的特点包括...
        """
        # 边界情况：如果没有历史记录
        if not self.chat_history:
            return "（这是第一轮对话）"

        # 格式化所有历史对话
        history_parts = []
        for i, (q, a) in enumerate(self.chat_history, 1):
            # enumerate(list, 1) 从 1 开始编号
            # (q, a) 元组解包：q=问题, a=答案
            history_parts.append(f"用户: {q}")
            history_parts.append(f"助手: {a}")

        # 用换行连接所有部分
        return "\n".join(history_parts)

    def _add_to_history(self, question: str, answer: str):
        """
        添加一轮对话到历史记录
        
        在每次查询完成后调用，将（问题, 答案）对保存到历史中。
        同时自动管理历史长度，防止无限增长。
        
        Args:
            question: 用户的问题
            answer: 助手的回答
        
        历史管理策略：
        - 新对话添加到列表末尾（append）
        - 如果超过 max_history 限制，删除最早的对话（FIFO）
        - 使用 pop(0) 删除第一个元素
        
        数据结构：
        - 使用元组 (question, answer) 存储一轮对话
        - 列表保持时间顺序：[旧对话, ..., 新对话]
        """
        # 添加新对话到历史末尾
        # 使用元组存储，不可变且高效
        self.chat_history.append((question, answer))

        # 保持历史在限制内（FIFO 队列策略）
        if len(self.chat_history) > self.max_history:
            # pop(0) 删除并返回第一个元素
            # 时间复杂度：O(n)，但对于小列表影响不大
            self.chat_history.pop(0)  # 移除最早的一条对话

    def stream_query(self, question: str, k: int = 4):
        """
        流式执行带历史的 RAG 查询
        
        与父类的区别：
        - 包含对话历史
        - 保存当前对话到历史
        
        Yields:
            生成的文本片段
        """
        try:
            # 检索文档
            relevant_docs = self.vectorstore.similarity_search(question, k=k)
            
            if not relevant_docs:
                answer = "抱歉，我在知识库中没有找到相关信息。"
                self._add_to_history(question, answer)
                yield answer
                return
            
            # 组装上下文
            context = "\n\n".join([
                f"[文档{i}]\n{doc.page_content}"
                for i, doc in enumerate(relevant_docs, 1)
            ])
            
            # 组装历史
            history = self._format_history()
            
            # 组装 Prompt（包含历史）
            prompt = self.prompt_template.format(
                history=history,
                context=context,
                question=question
            )
            
            # 流式生成并收集完整答案
            answer_parts = []
            for chunk in self.llm.stream_generate(prompt):
                answer_parts.append(chunk)
                yield chunk
            
            # 保存完整答案到历史
            full_answer = "".join(answer_parts)
            self._add_to_history(question, full_answer)
            
        except Exception as e:
            raise Exception(f"流式对话 RAG 查询失败: {str(e)}") from e
    
    def clear_history(self):
        """
        清空对话历史
        
        将对话历史列表重置为空，开始新的对话会话。
        
        使用场景：
        - 用户明确要求开始新对话
        - 对话主题完全切换
        - 历史信息不再相关
        - 测试或调试时重置状态
        
        注意事项：
        - 此操作不可逆，历史将永久丢失
        - 清空后，下次查询将被视为第一轮对话
        - 不影响向量库中的文档数据
        
        示例：
            >>> conv_chain = ConversationalRAGChain(llm, vectorstore)
            >>> conv_chain.query("Python 是什么？")
            >>> conv_chain.query("它有什么特点？")  # 能理解上下文
            >>> conv_chain.clear_history()  # 清空历史
            >>> conv_chain.query("它有什么特点？")  # 无法理解 "它"
        """
        # 重置历史列表为空
        # 下次 _format_history() 会返回 "（这是第一轮对话）"
        self.chat_history = []
