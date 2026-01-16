"""
LangGraph Agent 节点定义

节点是 LangGraph Agent 的核心组成部分，每个节点代表一个处理步骤。
节点接收 State，执行特定的业务逻辑，然后更新并返回 State。

核心概念：
1. 节点是纯函数：接收 State，返回更新后的 State
2. 节点职责单一：每个节点只做一件事
3. 节点可组合：多个节点组成完整的 Agent 流程

包含的节点：
- retrieve_node: 检索节点（从向量库检索相关文档）
- generate_node: 生成节点（基于文档生成答案）
- decide_node: 决策节点（决定下一步操作）
- rewrite_query_node: 查询改写节点（优化检索查询）
- evaluate_node: 评估节点（评估答案质量）
"""
from typing import Dict, Any, List, Optional
from langchain_core.documents import Document

# 导入项目模块
from src.agent.state import AgentState
from src.core.llm.base import BaseLLM
from src.core.vectorstore.base import BaseVectorStore


# ============================================================
# 核心节点
# ============================================================

def create_retrieve_node(vectorstore: BaseVectorStore, k: int = 4):
    """
    创建检索节点（工厂函数）
    
    为什么使用工厂函数？
    - 节点需要访问外部资源（vectorstore）
    - 直接在节点函数中硬编码会降低灵活性
    - 工厂函数允许在创建时注入依赖
    
    Args:
        vectorstore: 向量存储实例
        k: 检索的文档数量（默认 4）
        
    Returns:
        检索节点函数
        
    示例：
        >>> vectorstore = FAISSVectorStore(...)
        >>> retrieve_node = create_retrieve_node(vectorstore, k=4)
        >>> state = retrieve_node(state)
    """
    def retrieve_node(state: AgentState) -> AgentState:
        """
        检索节点：从向量库中检索相关文档
        
        工作流程：
        1. 从 State 中获取查询（优先使用 retrieval_query，否则使用 question）
        2. 使用向量存储进行相似度检索
        3. 计算检索质量分数
        4. 更新 State（添加检索结果和分数）
        5. 更新步骤计数
        
        Args:
            state: Agent 状态
            
        Returns:
            更新后的状态（包含 retrieved_docs 和 retrieval_score）
            
        异常处理：
        - 如果检索失败，记录错误到 state["error"]
        - 返回空列表，避免中断流程
        """
        try:
            # 步骤1: 获取查询文本
            # 优先使用改写后的查询，如果没有则使用原始问题
            query = state.get("retrieval_query") or state["question"]
            
            # 步骤2: 执行检索
            docs = vectorstore.similarity_search(query, k=k)
            
            # 步骤3: 计算检索质量分数
            # 如果文档有分数元数据，计算平均分；否则根据数量评估
            if docs and hasattr(docs[0], 'metadata') and 'score' in docs[0].metadata:
                scores = [doc.metadata.get('score', 0) for doc in docs]
                retrieval_score = sum(scores) / len(scores) if scores else 0.0
            else:
                # 根据检索到的文档数量评估质量
                retrieval_score = min(len(docs) / k, 1.0) if docs else 0.0
            
            # 步骤4: 更新 State
            state["retrieved_docs"] = docs
            state["retrieval_score"] = retrieval_score
            state["current_node"] = "retrieve"
            state["step_count"] = state.get("step_count", 0) + 1
            
            # 步骤5: 判断是否需要更多上下文
            # 如果检索质量低或文档数量不足，标记需要更多上下文
            state["need_more_context"] = (
                retrieval_score < 0.6 or len(docs) < k // 2
            )
            
            return state
            
        except Exception as e:
            # 异常处理：记录错误但不中断流程
            state["error"] = f"检索失败: {str(e)}"
            state["retrieved_docs"] = []
            state["retrieval_score"] = 0.0
            state["need_more_context"] = True
            return state
    
    return retrieve_node


def create_generate_node(llm: BaseLLM, prompt_template: Optional[str] = None):
    """
    创建生成节点（工厂函数）
    
    Args:
        llm: LLM 实例
        prompt_template: 自定义 Prompt 模板（可选）
                        必须包含 {context} 和 {question} 占位符
        
    Returns:
        生成节点函数
        
    示例：
        >>> llm = OpenAILLM(...)
        >>> generate_node = create_generate_node(llm)
        >>> state = generate_node(state)
    """
    # 获取默认 Prompt 模板
    default_template = """你是一位专业的智能助手。请基于以下上下文信息回答用户问题。

上下文信息：
{context}

用户问题：
{question}

要求：
1. 仅基于上下文信息回答，确保准确性
2. 如果上下文中没有相关信息，请明确说明
3. 回答要完整、清晰、结构化

答案："""
    
    template = prompt_template or default_template
    
    def generate_node(state: AgentState) -> AgentState:
        """
        生成节点：基于检索到的文档生成答案
        
        工作流程：
        1. 从 State 中获取问题和检索到的文档
        2. 将文档组装成上下文字符串
        3. 使用 Prompt 模板组合上下文和问题
        4. 调用 LLM 生成答案
        5. 更新 State
        
        Args:
            state: Agent 状态（必须包含 question 和 retrieved_docs）
            
        Returns:
            更新后的状态（包含 answer）
            
        异常处理：
        - 如果没有检索到文档，生成提示信息
        - 如果生成失败，记录错误
        """
        try:
            # 步骤1: 获取问题
            question = state["question"]
            
            # 步骤2: 获取并组装上下文
            docs = state.get("retrieved_docs", [])
            
            if not docs:
                # 没有检索到文档，返回提示信息
                state["answer"] = "抱歉，我在知识库中没有找到相关信息来回答您的问题。"
                state["confidence_score"] = 0.0
                state["current_node"] = "generate"
                state["step_count"] = state.get("step_count", 0) + 1
                return state
            
            # 组装上下文：将多个文档合并为一个字符串
            context = "\n\n".join([
                f"[文档 {i+1}]\n{doc.page_content}"
                for i, doc in enumerate(docs)
            ])
            
            # 步骤3: 组装 Prompt
            prompt = template.format(
                context=context,
                question=question
            )
            
            # 步骤4: 调用 LLM 生成答案
            answer = llm.generate(prompt)
            
            # 步骤5: 估算置信度
            # 简单策略：基于检索质量和答案长度
            retrieval_score = state.get("retrieval_score", 0.5)
            answer_length_score = min(len(answer) / 100, 1.0)  # 假设100字以上为完整答案
            confidence_score = (retrieval_score * 0.6 + answer_length_score * 0.4)
            
            # 步骤6: 更新 State
            state["answer"] = answer
            state["confidence_score"] = confidence_score
            state["current_node"] = "generate"
            state["step_count"] = state.get("step_count", 0) + 1
            
            return state
            
        except Exception as e:
            # 异常处理
            state["error"] = f"生成失败: {str(e)}"
            state["answer"] = f"抱歉，生成答案时出现错误：{str(e)}"
            state["confidence_score"] = 0.0
            return state
    
    return generate_node


def decide_node(state: AgentState) -> str:
    """
    决策节点：决定 Agent 的下一步操作
    
    这是一个条件边（Conditional Edge）使用的函数。
    与其他节点不同，决策节点返回的是字符串（下一个节点的名称），
    而不是更新后的 State。
    
    决策逻辑：
    1. 如果达到最大步数 → "end"（结束）
    2. 如果已有答案 → "end"（结束）
    3. 如果有错误 → "end"（结束）
    4. 如果需要更多上下文 → "retrieve"（重新检索）
    5. 如果还没有检索 → "retrieve"（开始检索）
    6. 如果检索完但还没有生成 → "generate"（生成答案）
    7. 默认 → "end"（结束）
    
    Args:
        state: Agent 状态
        
    Returns:
        下一个节点的名称（字符串）
        
    示例：
        >>> next_node = decide_node(state)
        >>> print(next_node)  # "retrieve", "generate", 或 "end"
    """
    # 决策1: 检查是否达到最大步数
    if state.get("step_count", 0) >= state.get("max_steps", 5):
        return "end"
    
    # 决策2: 检查是否已有答案
    if state.get("answer"):
        return "end"
    
    # 决策3: 检查是否有错误
    if state.get("error"):
        return "end"
    
    # 决策4: 检查是否需要更多上下文（但已经检索过）
    if state.get("need_more_context") and state.get("retrieved_docs") is not None:
        # 这里可以扩展：重写查询、增加检索数量等
        # 当前简化为直接生成
        return "generate"
    
    # 决策5: 检查是否还没有检索
    if state.get("retrieved_docs") is None:
        return "retrieve"
    
    # 决策6: 检索完但还没有生成
    if state.get("retrieved_docs") and not state.get("answer"):
        return "generate"
    
    # 默认：结束
    return "end"


# ============================================================
# 辅助节点（可选）
# ============================================================

def rewrite_query_node(state: AgentState) -> AgentState:
    """
    查询改写节点：优化检索查询
    
    应用场景：
    1. 代词消解：将"它"、"他"替换为具体实体
    2. 查询扩展：添加同义词或相关术语
    3. 查询简化：去除无关修饰词
    
    当前实现：简单的代词消解
    实际项目中可以使用 LLM 进行智能改写
    
    Args:
        state: Agent 状态
        
    Returns:
        更新后的状态（包含 retrieval_query）
    """
    question = state["question"]
    
    # 简单的代词消解逻辑
    # 实际项目中应该使用 LLM 或 NLP 工具
    if "它" in question or "他" in question or "她" in question:
        # 从对话历史中提取最近提到的实体
        messages = state.get("messages", [])
        if messages:
            # 简化示例：从最后一条消息中提取第一个名词
            # 实际应该使用 NER（命名实体识别）
            last_message = messages[-1].content if messages else ""
            # 这里简化处理，实际需要更复杂的逻辑
            state["retrieval_query"] = question  # 保持原样
        else:
            state["retrieval_query"] = question
    else:
        # 不需要改写，直接使用原始问题
        state["retrieval_query"] = question
    
    state["current_node"] = "rewrite_query"
    state["step_count"] = state.get("step_count", 0) + 1
    
    return state


def evaluate_node(state: AgentState) -> AgentState:
    """
    评估节点：评估生成答案的质量
    
    评估维度：
    1. 答案长度：太短可能不完整
    2. 检索质量：检索分数低，答案可靠性差
    3. 关键词匹配：答案是否包含问题中的关键词
    
    Args:
        state: Agent 状态
        
    Returns:
        更新后的状态（包含 confidence_score）
    """
    answer = state.get("answer", "")
    question = state.get("question", "")
    retrieval_score = state.get("retrieval_score", 0.5)
    
    # 评估1: 答案长度
    if len(answer) < 20:
        length_score = 0.3
    elif len(answer) < 100:
        length_score = 0.6
    else:
        length_score = 1.0
    
    # 评估2: 检索质量（已有）
    # retrieval_score 范围 0-1
    
    # 评估3: 关键词匹配（简化版）
    # 实际应该使用 TF-IDF 或语义相似度
    question_words = set(question.lower().split())
    answer_words = set(answer.lower().split())
    keyword_match = len(question_words & answer_words) / max(len(question_words), 1)
    
    # 综合评分
    confidence_score = (
        length_score * 0.3 +
        retrieval_score * 0.5 +
        keyword_match * 0.2
    )
    
    state["confidence_score"] = confidence_score
    state["current_node"] = "evaluate"
    state["step_count"] = state.get("step_count", 0) + 1
    
    return state


def should_continue(state: AgentState) -> str:
    """
    通用的继续判断函数
    
    用于条件边，决定是否继续执行或结束
    
    Args:
        state: Agent 状态
        
    Returns:
        "continue" 或 "end"
    """
    # 检查终止条件
    if state.get("step_count", 0) >= state.get("max_steps", 5):
        return "end"
    
    if state.get("answer"):
        return "end"
    
    if state.get("error"):
        return "end"
    
    return "continue"


# ============================================================
# 流式生成节点（高级）
# ============================================================

def create_stream_generate_node(llm: BaseLLM, prompt_template: Optional[str] = None):
    """
    创建流式生成节点
    
    注意：流式生成在 LangGraph 中需要特殊处理
    这里提供一个基础实现，实际使用时可能需要调整
    
    Args:
        llm: 支持流式生成的 LLM 实例
        prompt_template: Prompt 模板
        
    Returns:
        流式生成节点函数
    """
    default_template = """基于以下上下文回答问题：

上下文：
{context}

问题：{question}

答案："""
    
    template = prompt_template or default_template
    
    def stream_generate_node(state: AgentState) -> AgentState:
        """
        流式生成节点
        
        注意：这个实现会等待完整流式输出完成
        如果需要真正的流式返回，需要在 Graph 层面处理
        """
        try:
            question = state["question"]
            docs = state.get("retrieved_docs", [])
            
            if not docs:
                state["answer"] = "没有找到相关信息。"
                return state
            
            context = "\n\n".join([doc.page_content for doc in docs])
            prompt = template.format(context=context, question=question)
            
            # 收集流式输出
            answer_parts = []
            for chunk in llm.stream_generate(prompt):
                answer_parts.append(chunk)
                # 这里可以添加回调，实时输出
            
            state["answer"] = "".join(answer_parts)
            state["current_node"] = "stream_generate"
            state["step_count"] = state.get("step_count", 0) + 1
            
            return state
            
        except Exception as e:
            state["error"] = f"流式生成失败: {str(e)}"
            state["answer"] = "生成答案时出错。"
            return state
    
    return stream_generate_node


# ============================================================
# 导出
# ============================================================

__all__ = [
    # 工厂函数（推荐使用）
    "create_retrieve_node",
    "create_generate_node",
    "create_stream_generate_node",
    
    # 决策和判断函数
    "decide_node",
    "should_continue",
    
    # 辅助节点
    "rewrite_query_node",
    "evaluate_node",
]

