"""
LangGraph 图构建模块

这个模块定义了如何将节点（Nodes）连接成完整的 Agent 图（Graph）。
图定义了 Agent 的执行流程，包括节点的顺序、条件分支和循环。

核心概念：
1. Graph：节点和边的集合，定义了执行流程
2. Node：处理单元（已在 nodes.py 中定义）
3. Edge：节点间的连接（普通边和条件边）
4. Entry Point：执行的起点
5. END：特殊节点，表示执行结束

包含的 Agent 类型：
- SimpleRAGAgent：简单的 RAG Agent（retrieve → generate）
- ConditionalRAGAgent：带决策的 RAG Agent（decide → retrieve/generate）
- AdvancedRAGAgent：高级 Agent（查询改写、质量评估等）
- SelfReflectiveAgent：自我反思 Agent（生成 → 评估 → 改进）
"""
from typing import Callable, Dict, Any, Optional
from langgraph.graph import StateGraph, END

# 导入项目模块
from src.agent.state import AgentState
from src.agent.nodes import (
    create_retrieve_node,
    create_generate_node,
    decide_node,
    rewrite_query_node,
    evaluate_node,
    should_continue
)
from src.core.llm.base import BaseLLM
from src.core.vectorstore.base import BaseVectorStore


# ============================================================
# 基础 Agent：简单 RAG
# ============================================================

def create_simple_rag_agent(
    llm: BaseLLM,
    vectorstore: BaseVectorStore,
    k: int = 4
):
    """
    创建简单的 RAG Agent
    
    流程图：
    ```
    START → retrieve → generate → END
    ```
    
    工作流程：
    1. 从向量库检索相关文档
    2. 基于文档生成答案
    3. 结束
    
    适用场景：
    - 简单的问答系统
    - 单次检索即可回答的问题
    - 学习 LangGraph 的第一个 Agent
    
    Args:
        llm: LLM 实例
        vectorstore: 向量存储实例
        k: 检索的文档数量
        
    Returns:
        编译后的 Agent 图
        
    示例：
        >>> agent = create_simple_rag_agent(llm, vectorstore, k=4)
        >>> result = agent.invoke({"question": "Python是什么？"})
        >>> print(result["answer"])
    """
    # 创建节点
    retrieve_node = create_retrieve_node(vectorstore, k=k)
    generate_node = create_generate_node(llm)
    
    # 创建图
    graph = StateGraph(AgentState)
    
    # 添加节点
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    
    # 设置入口点
    graph.set_entry_point("retrieve")
    
    # 添加边：retrieve → generate → END
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    
    # 编译并返回
    return graph.compile()


# ============================================================
# 条件分支 Agent：带决策的 RAG
# ============================================================

def create_conditional_rag_agent(
    llm: BaseLLM,
    vectorstore: BaseVectorStore,
    k: int = 4
):
    """
    创建带决策的 RAG Agent
    
    流程图：
    ```
    START → decide → retrieve → decide → generate → decide → END
                 ↓                    ↓              ↓
                 └────────────────────┴──────────────┘
                         (条件分支)
    ```
    
    工作流程：
    1. 决策：判断当前应该做什么
    2. 根据决策执行相应节点（retrieve 或 generate）
    3. 重新决策，直到完成
    
    决策逻辑（decide_node）：
    - 未检索 → retrieve
    - 已检索但未生成 → generate
    - 已生成或达到最大步数 → END
    
    适用场景：
    - 需要多步骤推理
    - 可能需要多次检索
    - 需要根据中间结果调整流程
    
    Args:
        llm: LLM 实例
        vectorstore: 向量存储实例
        k: 检索的文档数量
        
    Returns:
        编译后的 Agent 图
        
    示例：
        >>> agent = create_conditional_rag_agent(llm, vectorstore, k=4)
        >>> state = create_initial_state("复杂问题", max_steps=5)
        >>> result = agent.invoke(state)
    """
    # 创建节点
    retrieve_node = create_retrieve_node(vectorstore, k=k)
    generate_node = create_generate_node(llm)
    
    # 创建图
    graph = StateGraph(AgentState)
    
    # 添加节点
    graph.add_node("decide", lambda state: state)  # 决策节点不修改 state
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    
    # 设置入口点
    graph.set_entry_point("decide")
    
    # 添加条件边：从 decide 出发
    graph.add_conditional_edges(
        "decide",           # 从哪个节点出发
        decide_node,        # 使用哪个决策函数
        {
            "retrieve": "retrieve",  # 如果返回 "retrieve"，跳到 retrieve 节点
            "generate": "generate",  # 如果返回 "generate"，跳到 generate 节点
            "end": END               # 如果返回 "end"，结束执行
        }
    )
    
    # 添加边：执行完节点后回到 decide
    graph.add_edge("retrieve", "decide")
    graph.add_edge("generate", "decide")
    
    # 编译并返回
    return graph.compile()


# ============================================================
# 高级 Agent：带查询改写和评估
# ============================================================

def create_advanced_rag_agent(
    llm: BaseLLM,
    vectorstore: BaseVectorStore,
    k: int = 4,
    enable_query_rewrite: bool = True,
    enable_evaluation: bool = True
):
    """
    创建高级 RAG Agent
    
    流程图：
    ```
    START → rewrite_query? → retrieve → evaluate_retrieval → decide
                                                                ↓
            ┌───────────────────────────────────────────────────┤
            ↓                                                   ↓
         generate → evaluate_answer → decide → END         retrieve
                                        ↓                   (重新检索)
                                       END
    ```
    
    工作流程：
    1. 查询改写（可选）：优化检索查询
    2. 检索文档
    3. 评估检索质量
    4. 决策：质量好 → 生成，质量差 → 重新检索
    5. 生成答案
    6. 评估答案质量（可选）
    7. 决策：完成 → END
    
    新增功能：
    - 查询改写（代词消解、查询扩展）
    - 检索质量评估
    - 答案质量评估
    - 基于质量的重试机制
    
    适用场景：
    - 生产环境部署
    - 需要高质量答案
    - 复杂问题需要多次尝试
    
    Args:
        llm: LLM 实例
        vectorstore: 向量存储实例
        k: 检索的文档数量
        enable_query_rewrite: 是否启用查询改写
        enable_evaluation: 是否启用质量评估
        
    Returns:
        编译后的 Agent 图
    """
    # 创建节点
    retrieve_node = create_retrieve_node(vectorstore, k=k)
    generate_node = create_generate_node(llm)
    
    # 创建图
    graph = StateGraph(AgentState)
    
    # 添加核心节点
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    
    # 添加可选节点
    if enable_query_rewrite:
        graph.add_node("rewrite_query", rewrite_query_node)
    
    if enable_evaluation:
        graph.add_node("evaluate", evaluate_node)
    
    # 添加决策节点
    graph.add_node("decide", lambda state: state)
    
    # 设置入口点
    if enable_query_rewrite:
        graph.set_entry_point("rewrite_query")
        graph.add_edge("rewrite_query", "retrieve")
    else:
        graph.set_entry_point("retrieve")
    
    # 构建流程
    graph.add_edge("retrieve", "decide")
    
    # 条件边：决策 → retrieve/generate/end
    graph.add_conditional_edges(
        "decide",
        decide_node,
        {
            "retrieve": "retrieve",
            "generate": "generate",
            "end": END
        }
    )
    
    # 生成后的处理
    if enable_evaluation:
        graph.add_edge("generate", "evaluate")
        graph.add_edge("evaluate", END)
    else:
        graph.add_edge("generate", END)
    
    return graph.compile()


# ============================================================
# 自我反思 Agent：生成-评估-改进循环
# ============================================================

def create_self_reflective_agent(
    llm: BaseLLM,
    vectorstore: BaseVectorStore,
    k: int = 4,
    max_refinements: int = 2
):
    """
    创建自我反思 RAG Agent
    
    流程图：
    ```
    START → retrieve → generate → evaluate → should_refine?
                           ↑                        ↓ yes
                           └────── refine ──────────┘
                                                    ↓ no
                                                   END
    ```
    
    工作流程：
    1. 检索文档
    2. 生成初步答案
    3. 评估答案质量
    4. 如果质量不够且未达到最大改进次数：
       - 基于评估反馈改进答案
       - 重新评估
    5. 否则：结束
    
    核心特性：
    - 自我评估能力
    - 迭代改进答案
    - 质量驱动的循环
    
    适用场景：
    - 需要高质量答案
    - 复杂问题需要多次打磨
    - 对答案质量有严格要求
    
    Args:
        llm: LLM 实例
        vectorstore: 向量存储实例
        k: 检索的文档数量
        max_refinements: 最大改进次数
        
    Returns:
        编译后的 Agent 图
    """
    # 创建节点
    retrieve_node = create_retrieve_node(vectorstore, k=k)
    generate_node = create_generate_node(llm)
    
    def refine_node(state: AgentState) -> AgentState:
        """改进节点：基于评估反馈改进答案"""
        current_answer = state.get("answer", "")
        confidence = state.get("confidence_score", 0)
        question = state["question"]
        docs = state.get("retrieved_docs", [])
        
        # 组装改进 prompt
        context = "\n\n".join([doc.page_content for doc in docs])
        refine_prompt = f"""
        原始问题：{question}
        
        当前答案：{current_answer}
        
        当前置信度：{confidence:.2f}
        
        上下文信息：{context}
        
        请改进以上答案，使其：
        1. 更加准确和完整
        2. 更加清晰和结构化
        3. 更好地利用上下文信息
        
        改进后的答案：
        """
        
        improved_answer = llm.generate(refine_prompt)
        
        # 更新 state
        state["answer"] = improved_answer
        state["refinement_count"] = state.get("refinement_count", 0) + 1
        state["step_count"] = state.get("step_count", 0) + 1
        
        return state
    
    def should_refine(state: AgentState) -> str:
        """决定是否需要改进答案"""
        confidence = state.get("confidence_score", 0)
        refinement_count = state.get("refinement_count", 0)
        
        # 如果置信度低且未达到最大改进次数
        if confidence < 0.8 and refinement_count < max_refinements:
            return "refine"
        else:
            return "end"
    
    # 创建图
    graph = StateGraph(AgentState)
    
    # 添加节点
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_node("evaluate", evaluate_node)
    graph.add_node("refine", refine_node)
    
    # 设置入口点
    graph.set_entry_point("retrieve")
    
    # 添加边
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "evaluate")
    graph.add_edge("refine", "evaluate")  # 改进后重新评估
    
    # 添加条件边：评估后决定是否改进
    graph.add_conditional_edges(
        "evaluate",
        should_refine,
        {
            "refine": "refine",
            "end": END
        }
    )
    
    return graph.compile()


# ============================================================
# 通用 Agent 构建器
# ============================================================

def create_custom_agent(
    nodes: Dict[str, Callable],
    edges: list,
    conditional_edges: Optional[list] = None,
    entry_point: str = "start"
):
    """
    通用 Agent 构建器：自定义节点和边
    
    这是一个灵活的构建器，允许你完全自定义 Agent 的结构。
    
    Args:
        nodes: 节点字典 {"节点名": 节点函数}
        edges: 边列表 [("from_node", "to_node"), ...]
        conditional_edges: 条件边列表 [
            ("from_node", decide_func, {"option1": "node1", "option2": "node2"}),
            ...
        ]
        entry_point: 入口节点名称
        
    Returns:
        编译后的 Agent 图
        
    示例：
        >>> # 定义节点
        >>> nodes = {
        ...     "retrieve": retrieve_node,
        ...     "generate": generate_node,
        ...     "decide": lambda s: s
        ... }
        >>> 
        >>> # 定义边
        >>> edges = [("retrieve", "generate")]
        >>> 
        >>> # 定义条件边
        >>> conditional_edges = [
        ...     ("decide", decide_node, {
        ...         "retrieve": "retrieve",
        ...         "end": END
        ...     })
        ... ]
        >>> 
        >>> # 创建 Agent
        >>> agent = create_custom_agent(
        ...     nodes=nodes,
        ...     edges=edges,
        ...     conditional_edges=conditional_edges,
        ...     entry_point="decide"
        ... )
    """
    # 创建图
    graph = StateGraph(AgentState)
    
    # 添加所有节点
    for node_name, node_func in nodes.items():
        graph.add_node(node_name, node_func)
    
    # 设置入口点
    graph.set_entry_point(entry_point)
    
    # 添加普通边
    for from_node, to_node in edges:
        graph.add_edge(from_node, to_node)
    
    # 添加条件边
    if conditional_edges:
        for from_node, decide_func, mapping in conditional_edges:
            graph.add_conditional_edges(from_node, decide_func, mapping)
    
    return graph.compile()


# ============================================================
# Agent 工厂函数
# ============================================================

def create_agent(
    agent_type: str,
    llm: BaseLLM,
    vectorstore: BaseVectorStore,
    **kwargs
):
    """
    Agent 工厂函数：根据类型创建 Agent
    
    这是一个便捷函数，根据字符串类型创建对应的 Agent。
    
    Args:
        agent_type: Agent 类型
            - "simple": 简单 RAG Agent
            - "conditional": 带决策的 RAG Agent
            - "advanced": 高级 RAG Agent
            - "self_reflective": 自我反思 Agent
        llm: LLM 实例
        vectorstore: 向量存储实例
        **kwargs: 传递给具体 Agent 创建函数的参数
        
    Returns:
        编译后的 Agent 图
        
    Raises:
        ValueError: 不支持的 Agent 类型
        
    示例：
        >>> # 创建简单 Agent
        >>> agent = create_agent("simple", llm, vectorstore, k=4)
        >>> 
        >>> # 创建高级 Agent
        >>> agent = create_agent(
        ...     "advanced",
        ...     llm,
        ...     vectorstore,
        ...     k=4,
        ...     enable_query_rewrite=True
        ... )
    """
    agent_types = {
        "simple": create_simple_rag_agent,
        "conditional": create_conditional_rag_agent,
        "advanced": create_advanced_rag_agent,
        "self_reflective": create_self_reflective_agent
    }
    
    if agent_type not in agent_types:
        raise ValueError(
            f"不支持的 Agent 类型: {agent_type}。"
            f"支持的类型: {list(agent_types.keys())}"
        )
    
    creator_func = agent_types[agent_type]
    return creator_func(llm, vectorstore, **kwargs)


# ============================================================
# 辅助函数
# ============================================================

def visualize_graph(graph):
    """
    可视化 Agent 图（需要 graphviz）
    
    Args:
        graph: 编译后的 Agent 图
        
    Returns:
        图的可视化表示（如果可用）
        
    注意：
        需要安装 graphviz: pip install graphviz
    """
    try:
        # LangGraph 提供的可视化方法
        return graph.get_graph().draw_mermaid()
    except Exception as e:
        return f"可视化失败: {str(e)}\n提示：需要安装 graphviz"


def get_graph_info(graph) -> Dict[str, Any]:
    """
    获取图的信息
    
    Args:
        graph: 编译后的 Agent 图
        
    Returns:
        图的信息字典
    """
    try:
        graph_obj = graph.get_graph()
        
        return {
            "nodes": list(graph_obj.nodes.keys()),
            "edges": [
                (edge.source, edge.target)
                for edge in graph_obj.edges
            ],
            "entry_point": graph_obj.entry_point,
            "node_count": len(graph_obj.nodes),
            "edge_count": len(graph_obj.edges)
        }
    except Exception as e:
        return {"error": str(e)}


# ============================================================
# 导出
# ============================================================

__all__ = [
    # 基础 Agent
    "create_simple_rag_agent",
    
    # 条件分支 Agent
    "create_conditional_rag_agent",
    
    # 高级 Agent
    "create_advanced_rag_agent",
    
    # 自我反思 Agent
    "create_self_reflective_agent",
    
    # 通用构建器
    "create_custom_agent",
    
    # 工厂函数
    "create_agent",
    
    # 辅助函数
    "visualize_graph",
    "get_graph_info",
]

