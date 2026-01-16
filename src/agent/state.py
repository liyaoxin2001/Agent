"""
LangGraph Agent 状态定义

这个模块定义了 Agent 执行过程中的状态结构。
State 是所有节点共享的"工作空间"，用于传递数据和记录执行进度。

设计理念：
1. 从简单开始 - 只包含必要的字段
2. 类型安全 - 使用 TypedDict 提供类型检查
3. 可扩展 - 根据需求逐步添加字段

参考文档：https://langchain-ai.github.io/langgraph/concepts/low_level/#state
"""
from typing import TypedDict, List, Optional, Dict, Any
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage


# ============================================================
# 版本1: 基础版本 - 适合初学者
# ============================================================

class AgentStateBasic(TypedDict):
    """
    Agent 状态 - 基础版本
    
    适用场景：
    - 简单的 RAG 流程（问题 → 检索 → 生成）
    - 学习 LangGraph 的第一个 Agent
    - 快速原型开发
    
    工作流程：
    1. 初始化：question 有值，其他为 None
    2. 检索节点：填充 retrieved_docs
    3. 生成节点：填充 answer
    """
    
    question: str
    """用户提出的问题"""
    
    retrieved_docs: Optional[List[Document]]
    """从向量库检索到的相关文档列表"""
    
    answer: Optional[str]
    """LLM 生成的最终答案"""


# ============================================================
# 版本2: 进阶版本 - 支持多轮对话
# ============================================================

class AgentStateConversational(TypedDict):
    """
    Agent 状态 - 对话版本
    
    适用场景：
    - 需要多轮对话
    - 需要查询改写（代词消解、查询扩展）
    - 需要控制执行流程
    
    新增功能：
    - messages: 对话历史管理
    - retrieval_query: 支持查询改写
    - step_count: 执行步骤控制
    """
    
    # ========== 核心字段 ==========
    question: str
    """用户当前问题"""
    
    retrieved_docs: Optional[List[Document]]
    """检索到的相关文档"""
    
    answer: Optional[str]
    """生成的答案"""
    
    # ========== 对话管理 ==========
    messages: List[BaseMessage]
    """完整的对话历史 [HumanMessage, AIMessage, ...]"""
    
    retrieval_query: Optional[str]
    """实际用于检索的查询（可能经过改写）
    
    示例：
    - question: "它的应用领域有哪些？"
    - retrieval_query: "Python 的应用领域有哪些？"（代词替换）
    """
    
    # ========== 执行控制 ==========
    step_count: int
    """当前执行的步骤数（从 0 开始）"""
    
    max_steps: int
    """允许的最大步骤数（防止无限循环）"""


# ============================================================
# 版本3: 完整版本 - 生产级（默认使用）
# ============================================================

class AgentState(TypedDict):
    """
    Agent 状态 - 完整版本
    
    这是推荐在生产环境使用的完整状态定义。
    包含了调试、监控、工具调用等高级功能。
    
    适用场景：
    - 生产环境部署
    - 需要详细日志和监控
    - 复杂的多步骤 Agent
    - 需要工具调用
    
    使用示例：
    ```python
    # 初始化
    state = create_initial_state(
        question="Python 的应用领域有哪些？",
        max_steps=5
    )
    
    # 在节点中更新
    def retrieve_node(state):
        docs = vectorstore.search(state["question"])
        state["retrieved_docs"] = docs
        state["step_count"] += 1
        return state
    ```
    """
    
    # ========== 核心字段 ==========
    question: str
    """用户提出的原始问题"""
    
    answer: Optional[str]
    """LLM 生成的最终答案"""
    
    # ========== 检索相关 ==========
    retrieved_docs: Optional[List[Document]]
    """从向量库检索到的文档列表
    
    每个 Document 包含：
    - page_content: 文档内容
    - metadata: 元数据（来源、页码等）
    """
    
    retrieval_query: Optional[str]
    """经过改写/优化后的检索查询
    
    应用场景：
    1. 代词消解：将"它"替换为具体实体
    2. 查询扩展：添加同义词或相关术语
    3. 查询简化：去除无关的修饰词
    """
    
    retrieval_score: Optional[float]
    """检索质量评分（0.0 - 1.0）
    
    用途：
    - 判断是否需要重新检索
    - 决定是否需要更多上下文
    - 记录检索效果用于优化
    """
    
    need_more_context: bool
    """是否需要更多上下文信息
    
    触发条件：
    - 检索文档太少
    - 文档相关性低
    - 问题过于宽泛
    """
    
    # ========== 生成相关 ==========
    intermediate_answer: Optional[str]
    """中间答案（用于多步推理）
    
    示例：
    - 第一步：生成初步答案
    - 第二步：基于初步答案补充细节
    - 第三步：综合形成最终答案
    """
    
    confidence_score: Optional[float]
    """答案置信度评分（0.0 - 1.0）
    
    用途：
    - 低置信度时提示用户或重试
    - 记录答案质量
    - A/B 测试对比
    """
    
    # ========== 对话管理 ==========
    messages: List[BaseMessage]
    """完整的对话历史记录
    
    结构：
    [
        HumanMessage(content="Python是什么？"),
        AIMessage(content="Python是..."),
        HumanMessage(content="它的应用领域有哪些？"),
        AIMessage(content="Python广泛应用于...")
    ]
    """
    
    conversation_id: Optional[str]
    """会话ID（用于多轮对话追踪）"""
    
    # ========== 执行控制 ==========
    step_count: int
    """当前执行的步骤数（从 0 开始）
    
    用于：
    - 防止无限循环
    - 调试和日志
    - 性能监控
    """
    
    max_steps: int
    """允许的最大执行步数（默认 5-10）
    
    超过此步数将强制终止，防止：
    - 无限循环
    - 资源耗尽
    - 用户等待过久
    """
    
    current_node: Optional[str]
    """当前执行的节点名称（用于日志和调试）
    
    示例值：
    - "retrieve"
    - "generate"  
    - "decide"
    - "tool_call"
    """
    
    next_action: Optional[str]
    """下一步要执行的动作（用于条件分支）
    
    示例：
    - "retrieve_more" - 需要更多检索
    - "generate" - 可以生成答案
    - "use_tool" - 需要调用工具
    - "end" - 结束执行
    """
    
    # ========== 工具调用（可选）==========
    tool_calls: Optional[List[Dict[str, Any]]]
    """工具调用记录
    
    结构：
    [
        {
            "tool": "web_search",
            "query": "Python 最新版本",
            "timestamp": "2026-01-13 10:30:00"
        }
    ]
    """
    
    tool_results: Optional[List[Any]]
    """工具执行结果
    
    与 tool_calls 一一对应
    """
    
    # ========== 元数据 ==========
    metadata: Dict[str, Any]
    """额外的元数据信息
    
    可以存储：
    - 用户信息
    - 会话配置
    - 自定义参数
    - 实验标记
    """
    
    error: Optional[str]
    """错误信息（如果执行过程中出错）
    
    格式：包含错误类型和详细描述
    示例："检索失败: 向量库连接超时"
    """


# ============================================================
# 辅助函数
# ============================================================

def create_initial_state(
    question: str,
    max_steps: int = 5,
    conversation_id: Optional[str] = None,
    messages: Optional[List[BaseMessage]] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> AgentState:
    """
    创建初始 Agent 状态
    
    这是推荐的状态初始化方式，确保所有必填字段都有正确的默认值。
    
    Args:
        question: 用户提出的问题
        max_steps: 允许的最大执行步数（默认 5）
        conversation_id: 会话ID（可选，用于多轮对话追踪）
        messages: 历史对话消息（可选，用于继续之前的会话）
        metadata: 额外的元数据（可选）
        
    Returns:
        初始化完成的 AgentState
        
    示例：
        >>> # 单轮对话
        >>> state = create_initial_state("Python是什么？")
        >>> 
        >>> # 多轮对话
        >>> state = create_initial_state(
        ...     question="它的应用领域有哪些？",
        ...     conversation_id="conv-123",
        ...     messages=[
        ...         HumanMessage(content="Python是什么？"),
        ...         AIMessage(content="Python是一种编程语言...")
        ...     ]
        ... )
    """
    return AgentState(
        # 核心字段
        question=question,
        answer=None,
        
        # 检索相关
        retrieved_docs=None,
        retrieval_query=None,
        retrieval_score=None,
        need_more_context=False,
        
        # 生成相关
        intermediate_answer=None,
        confidence_score=None,
        
        # 对话管理
        messages=messages or [],
        conversation_id=conversation_id,
        
        # 执行控制
        step_count=0,
        max_steps=max_steps,
        current_node=None,
        next_action=None,
        
        # 工具调用
        tool_calls=None,
        tool_results=None,
        
        # 元数据
        metadata=metadata or {},
        error=None
    )


def create_basic_state(question: str) -> AgentStateBasic:
    """
    创建基础版本的 Agent 状态
    
    适合初学者和简单场景。
    
    Args:
        question: 用户问题
        
    Returns:
        基础版本的 AgentState
        
    示例：
        >>> state = create_basic_state("Python是什么？")
        >>> print(state)
        {
            'question': 'Python是什么？',
            'retrieved_docs': None,
            'answer': None
        }
    """
    return AgentStateBasic(
        question=question,
        retrieved_docs=None,
        answer=None
    )


def create_conversational_state(
    question: str,
    messages: Optional[List[BaseMessage]] = None,
    max_steps: int = 5
) -> AgentStateConversational:
    """
    创建对话版本的 Agent 状态
    
    适合需要多轮对话的场景。
    
    Args:
        question: 用户问题
        messages: 历史对话消息
        max_steps: 最大执行步数
        
    Returns:
        对话版本的 AgentState
    """
    return AgentStateConversational(
        question=question,
        retrieved_docs=None,
        answer=None,
        messages=messages or [],
        retrieval_query=None,
        step_count=0,
        max_steps=max_steps
    )


# ============================================================
# 导出
# ============================================================

__all__ = [
    # 状态类（三个版本）
    "AgentState",                    # 完整版本（推荐）
    "AgentStateBasic",              # 基础版本
    "AgentStateConversational",     # 对话版本
    
    # 辅助函数
    "create_initial_state",         # 创建完整版本状态
    "create_basic_state",           # 创建基础版本状态
    "create_conversational_state",  # 创建对话版本状态
]

