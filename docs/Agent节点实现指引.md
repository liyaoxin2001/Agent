# Agent 节点实现指引

## 📋 文档概述

本文档详细说明 LangGraph Agent 节点的实现方法、设计原则和使用方式。

**目标读者**：学习 LangGraph Agent 开发的开发者

**前置知识**：需要先理解 AgentState 的概念（参考 `AgentState模块知识点总结.md`）

---

## 一、什么是节点（Node）？

### 1.1 基本概念

**定义**：节点是 LangGraph Agent 中的一个处理单元，负责执行特定的业务逻辑。

**本质**：节点是一个**纯函数**：
```
输入：AgentState
处理：执行业务逻辑
输出：更新后的 AgentState
```

**类比理解**：
- 节点就像**流水线上的工位**
- State 是**工作台上的产品**
- 每个工位做一件事，然后传给下一个工位

```
State (初始)
    ↓
[节点1: 检索] → State (添加了 retrieved_docs)
    ↓
[节点2: 生成] → State (添加了 answer)
    ↓
State (最终)
```

---

### 1.2 节点的特点

#### 特点1：函数签名

```python
def node_name(state: AgentState) -> AgentState:
    """节点函数"""
    # 1. 读取 State
    data = state["some_field"]
    
    # 2. 执行业务逻辑
    result = process(data)
    
    # 3. 更新 State
    state["output_field"] = result
    
    # 4. 返回 State
    return state
```

**关键点**：
- 输入：`AgentState` 类型
- 输出：`AgentState` 类型
- 无副作用：只通过 State 交互

#### 特点2：职责单一

```python
# ✅ 好的节点 - 职责单一
def retrieve_node(state):
    """只做检索"""
    docs = vectorstore.search(state["question"])
    state["retrieved_docs"] = docs
    return state

# ❌ 不好的节点 - 职责混乱
def retrieve_and_generate_node(state):
    """既检索又生成，职责不清"""
    docs = vectorstore.search(state["question"])
    answer = llm.generate(docs)
    state["answer"] = answer
    return state
```

**为什么要职责单一？**
- 便于测试
- 便于复用
- 便于调试
- 清晰的执行流程

#### 特点3：可组合

```python
# 节点可以像积木一样组合
state = initial_state

state = retrieve_node(state)    # 节点1
state = rewrite_node(state)     # 节点2
state = retrieve_node(state)    # 再次使用节点1
state = generate_node(state)    # 节点3

# 形成完整的 Agent 流程
```

---

## 二、核心节点实现

### 2.1 检索节点 - retrieve_node

#### 作用

从向量数据库中检索相关文档

#### 工作流程

```
1. 从 State 获取查询（question 或 retrieval_query）
2. 调用 vectorstore.similarity_search() 检索
3. 计算检索质量分数
4. 更新 State（retrieved_docs, retrieval_score）
5. 返回 State
```

#### 实现代码

```python
def create_retrieve_node(vectorstore: BaseVectorStore, k: int = 4):
    """
    创建检索节点（工厂函数）
    
    为什么使用工厂函数？
    - 节点需要访问外部资源（vectorstore）
    - 工厂函数允许在创建时注入依赖
    """
    def retrieve_node(state: AgentState) -> AgentState:
        # 1. 获取查询
        query = state.get("retrieval_query") or state["question"]
        
        # 2. 执行检索
        docs = vectorstore.similarity_search(query, k=k)
        
        # 3. 计算质量分数
        if docs and 'score' in docs[0].metadata:
            scores = [doc.metadata.get('score', 0) for doc in docs]
            retrieval_score = sum(scores) / len(scores)
        else:
            retrieval_score = min(len(docs) / k, 1.0)
        
        # 4. 更新 State
        state["retrieved_docs"] = docs
        state["retrieval_score"] = retrieval_score
        state["need_more_context"] = retrieval_score < 0.6
        state["step_count"] += 1
        
        return state
    
    return retrieve_node
```

#### 关键点解析

**1. 为什么使用工厂函数？**

```python
# ❌ 直接定义 - 无法注入依赖
def retrieve_node(state):
    # vectorstore 从哪里来？
    docs = vectorstore.search(state["question"])
    return state

# ✅ 工厂函数 - 可以注入依赖
def create_retrieve_node(vectorstore, k=4):
    def retrieve_node(state):
        docs = vectorstore.search(state["question"], k=k)
        return state
    return retrieve_node

# 使用
my_vectorstore = FAISSVectorStore(...)
retrieve_node = create_retrieve_node(my_vectorstore, k=4)
```

**2. 优先使用改写查询**

```python
# 为什么这样写？
query = state.get("retrieval_query") or state["question"]

# 场景1: 第一次检索，没有改写
state = {"question": "Python是什么？", "retrieval_query": None}
query = None or "Python是什么？"  # 使用原始问题

# 场景2: 查询已改写
state = {
    "question": "它的应用领域有哪些？",
    "retrieval_query": "Python的应用领域有哪些？"  # 代词已替换
}
query = "Python的应用领域有哪些？"  # 使用改写后的查询
```

**3. 质量分数计算**

```python
# 方法1: 使用文档自带分数（如果有）
if docs and 'score' in docs[0].metadata:
    scores = [doc.metadata.get('score', 0) for doc in docs]
    retrieval_score = sum(scores) / len(scores)

# 方法2: 基于数量评估（简单有效）
else:
    # 如果期望检索4个，实际检索到4个，分数=1.0
    # 如果期望4个，实际只有2个，分数=0.5
    retrieval_score = min(len(docs) / k, 1.0)
```

**4. 判断是否需要更多上下文**

```python
# 如果检索质量低，标记需要更多上下文
state["need_more_context"] = retrieval_score < 0.6

# 这个标记可以用于决策节点
def decide_node(state):
    if state["need_more_context"]:
        return "rewrite_query"  # 改写查询重新检索
    else:
        return "generate"  # 直接生成
```

---

### 2.2 生成节点 - generate_node

#### 作用

基于检索到的文档生成答案

#### 工作流程

```
1. 从 State 获取问题和文档
2. 组装文档为上下文字符串
3. 使用 Prompt 模板组合上下文和问题
4. 调用 LLM 生成答案
5. 计算置信度分数
6. 更新 State（answer, confidence_score）
7. 返回 State
```

#### 实现代码

```python
def create_generate_node(llm: BaseLLM, prompt_template: Optional[str] = None):
    """创建生成节点"""
    
    # 默认 Prompt 模板
    default_template = """请基于以下上下文信息回答问题：

上下文信息：
{context}

用户问题：
{question}

答案："""
    
    template = prompt_template or default_template
    
    def generate_node(state: AgentState) -> AgentState:
        # 1. 获取数据
        question = state["question"]
        docs = state.get("retrieved_docs", [])
        
        # 2. 检查是否有文档
        if not docs:
            state["answer"] = "抱歉，我没有找到相关信息。"
            state["confidence_score"] = 0.0
            return state
        
        # 3. 组装上下文
        context = "\n\n".join([
            f"[文档 {i+1}]\n{doc.page_content}"
            for i, doc in enumerate(docs)
        ])
        
        # 4. 组装 Prompt
        prompt = template.format(
            context=context,
            question=question
        )
        
        # 5. 生成答案
        answer = llm.generate(prompt)
        
        # 6. 计算置信度
        retrieval_score = state.get("retrieval_score", 0.5)
        answer_length_score = min(len(answer) / 100, 1.0)
        confidence_score = retrieval_score * 0.6 + answer_length_score * 0.4
        
        # 7. 更新 State
        state["answer"] = answer
        state["confidence_score"] = confidence_score
        state["step_count"] += 1
        
        return state
    
    return generate_node
```

#### 关键点解析

**1. 处理没有文档的情况**

```python
if not docs:
    # 不要抛出异常，而是返回友好提示
    state["answer"] = "抱歉，我没有找到相关信息。"
    state["confidence_score"] = 0.0
    return state  # 提前返回
```

**2. 上下文组装**

```python
# 方式1: 简单拼接
context = "\n\n".join([doc.page_content for doc in docs])

# 方式2: 添加编号（推荐）
context = "\n\n".join([
    f"[文档 {i+1}]\n{doc.page_content}"
    for i, doc in enumerate(docs)
])

# 方式3: 添加来源
context = "\n\n".join([
    f"[文档 {i+1} - {doc.metadata.get('source', '未知')}]\n{doc.page_content}"
    for i, doc in enumerate(docs)
])
```

**3. Prompt 设计**

```python
# 基础版
template = """上下文：{context}\n问题：{question}\n答案："""

# 专业版（推荐）
template = """你是一位专业助手。请基于以下上下文回答问题。

上下文信息：
{context}

用户问题：
{question}

要求：
1. 仅基于上下文信息回答
2. 如果上下文中没有相关信息，请明确说明
3. 回答要完整、准确、结构化

答案："""
```

**4. 置信度计算**

```python
# 综合多个因素
retrieval_score = state.get("retrieval_score", 0.5)  # 检索质量
answer_length_score = min(len(answer) / 100, 1.0)    # 答案长度

# 加权平均
confidence_score = (
    retrieval_score * 0.6 +      # 检索质量占60%
    answer_length_score * 0.4    # 答案长度占40%
)

# 更复杂的计算（可选）
keyword_match = calculate_keyword_match(question, answer)
confidence_score = (
    retrieval_score * 0.5 +
    answer_length_score * 0.3 +
    keyword_match * 0.2
)
```

---

### 2.3 决策节点 - decide_node

#### 作用

决定 Agent 的下一步操作

#### 特殊性

**决策节点与其他节点不同**：
- 其他节点：返回 `AgentState`
- 决策节点：返回 `str`（下一个节点的名称）

```python
# 普通节点
def retrieve_node(state: AgentState) -> AgentState:
    return state

# 决策节点
def decide_node(state: AgentState) -> str:
    return "next_node_name"  # 返回字符串！
```

#### 实现代码

```python
def decide_node(state: AgentState) -> str:
    """决策节点：决定下一步操作"""
    
    # 决策1: 检查是否达到最大步数
    if state.get("step_count", 0) >= state.get("max_steps", 5):
        return "end"  # 终止
    
    # 决策2: 检查是否已有答案
    if state.get("answer"):
        return "end"  # 完成
    
    # 决策3: 检查是否有错误
    if state.get("error"):
        return "end"  # 错误终止
    
    # 决策4: 还没有检索
    if state.get("retrieved_docs") is None:
        return "retrieve"  # 开始检索
    
    # 决策5: 已检索但质量差
    if state.get("need_more_context"):
        return "generate"  # 尝试生成（或 rewrite_query）
    
    # 决策6: 已检索但还没生成
    if state.get("retrieved_docs") and not state.get("answer"):
        return "generate"  # 生成答案
    
    # 默认：结束
    return "end"
```

#### 决策逻辑设计

**决策树**：
```
开始
  ├─ 步数超限？ → end
  ├─ 有答案？ → end
  ├─ 有错误？ → end
  ├─ 未检索？ → retrieve
  ├─ 质量差？ → rewrite_query 或 generate
  ├─ 未生成？ → generate
  └─ 默认 → end
```

**使用场景**：

```python
# 在 Graph 中使用决策节点
from langgraph.graph import StateGraph, END

graph = StateGraph(AgentState)

# 添加条件边
graph.add_conditional_edges(
    "decide",  # 从 decide 节点出发
    decide_node,  # 使用决策函数
    {
        "retrieve": "retrieve",  # 如果返回 "retrieve"，跳到 retrieve 节点
        "generate": "generate",  # 如果返回 "generate"，跳到 generate 节点
        "end": END               # 如果返回 "end"，结束执行
    }
)
```

---

## 三、辅助节点

### 3.1 查询改写节点

```python
def rewrite_query_node(state: AgentState) -> AgentState:
    """查询改写节点：优化检索查询"""
    
    question = state["question"]
    
    # 简单的代词消解
    if "它" in question or "他" in question:
        # 从对话历史中提取实体
        messages = state.get("messages", [])
        # ... 代词替换逻辑
        state["retrieval_query"] = processed_question
    else:
        state["retrieval_query"] = question
    
    return state
```

### 3.2 评估节点

```python
def evaluate_node(state: AgentState) -> AgentState:
    """评估节点：评估答案质量"""
    
    answer = state.get("answer", "")
    question = state.get("question", "")
    
    # 评估维度
    length_score = min(len(answer) / 100, 1.0)
    retrieval_score = state.get("retrieval_score", 0.5)
    keyword_match = calculate_match(question, answer)
    
    # 综合评分
    confidence_score = (
        length_score * 0.3 +
        retrieval_score * 0.5 +
        keyword_match * 0.2
    )
    
    state["confidence_score"] = confidence_score
    return state
```

---

## 四、节点设计模式

### 4.1 工厂模式（推荐）

**为什么使用工厂模式？**
- 节点需要外部依赖（LLM, VectorStore等）
- 配置参数化（k, temperature等）
- 便于测试和复用

```python
# 工厂函数
def create_retrieve_node(vectorstore, k=4):
    def retrieve_node(state):
        docs = vectorstore.search(state["question"], k=k)
        state["retrieved_docs"] = docs
        return state
    return retrieve_node

# 使用
vectorstore = FAISSVectorStore(...)
retrieve_node = create_retrieve_node(vectorstore, k=4)

# 测试时可以注入 mock
mock_vectorstore = MockVectorStore()
test_retrieve_node = create_retrieve_node(mock_vectorstore)
```

### 4.2 类方法模式（可选）

```python
class AgentNodes:
    """节点集合类"""
    
    def __init__(self, llm, vectorstore):
        self.llm = llm
        self.vectorstore = vectorstore
    
    def retrieve(self, state):
        docs = self.vectorstore.search(state["question"])
        state["retrieved_docs"] = docs
        return state
    
    def generate(self, state):
        answer = self.llm.generate(...)
        state["answer"] = answer
        return state

# 使用
nodes = AgentNodes(llm, vectorstore)
state = nodes.retrieve(state)
state = nodes.generate(state)
```

### 4.3 装饰器模式（高级）

```python
def with_error_handling(node_func):
    """添加错误处理的装饰器"""
    def wrapper(state):
        try:
            return node_func(state)
        except Exception as e:
            state["error"] = str(e)
            return state
    return wrapper

def with_logging(node_name):
    """添加日志的装饰器"""
    def decorator(node_func):
        def wrapper(state):
            print(f"[{node_name}] 开始执行...")
            result = node_func(state)
            print(f"[{node_name}] 执行完成")
            return result
        return wrapper
    return decorator

# 使用
@with_logging("retrieve")
@with_error_handling
def retrieve_node(state):
    # ... 实现
    return state
```

---

## 五、最佳实践

### 5.1 错误处理

```python
def retrieve_node(state):
    try:
        # 业务逻辑
        docs = vectorstore.search(state["question"])
        state["retrieved_docs"] = docs
    except Exception as e:
        # 记录错误但不中断流程
        state["error"] = f"检索失败: {str(e)}"
        state["retrieved_docs"] = []  # 提供默认值
    
    return state
```

### 5.2 日志记录

```python
def retrieve_node(state):
    # 记录开始
    state.setdefault("processing_log", []).append(
        f"[retrieve] 开始检索问题: {state['question']}"
    )
    
    # 执行业务逻辑
    docs = vectorstore.search(state["question"])
    
    # 记录结果
    state["processing_log"].append(
        f"[retrieve] 检索到 {len(docs)} 个文档"
    )
    
    return state
```

### 5.3 步骤计数

```python
def any_node(state):
    # 业务逻辑
    # ...
    
    # 更新步骤计数
    state["step_count"] = state.get("step_count", 0) + 1
    
    return state
```

### 5.4 节点标记

```python
def retrieve_node(state):
    # 标记当前节点（用于调试）
    state["current_node"] = "retrieve"
    
    # 业务逻辑
    # ...
    
    return state
```

---

## 六、常见问题

### Q1: 节点可以调用其他节点吗？

**A**: 不推荐，应该由 Graph 控制流程

```python
# ❌ 不推荐 - 节点间直接调用
def combined_node(state):
    state = retrieve_node(state)  # 调用另一个节点
    state = generate_node(state)
    return state

# ✅ 推荐 - 由 Graph 控制
graph.add_edge("retrieve", "generate")
```

### Q2: 节点必须返回 State 吗？

**A**: 普通节点必须返回 State，决策节点返回字符串

```python
# 普通节点 - 返回 State
def normal_node(state):
    return state

# 决策节点 - 返回字符串
def decide_node(state):
    return "next_node"
```

### Q3: 可以只返回部分更新吗？

**A**: 可以，LangGraph 会自动合并

```python
# 方式1: 返回完整 State
def node1(state):
    state["answer"] = "..."
    return state

# 方式2: 只返回更新字段
def node2(state):
    return {"answer": "..."}  # 自动合并
```

---

## 七、总结

### 核心要点

1. **节点是纯函数**：输入 State，输出 State
2. **职责单一**：每个节点只做一件事
3. **使用工厂模式**：注入外部依赖
4. **完善错误处理**：不要让异常中断流程
5. **记录日志和步骤**：便于调试和监控

### 节点清单

| 节点 | 作用 | 输入 | 输出 |
|------|------|------|------|
| retrieve_node | 检索文档 | question | retrieved_docs |
| generate_node | 生成答案 | retrieved_docs | answer |
| decide_node | 决策分支 | state | 节点名称(str) |
| rewrite_query_node | 改写查询 | question | retrieval_query |
| evaluate_node | 评估质量 | answer | confidence_score |

### 下一步

完成节点实现后，你将学习：
1. **图构建**（`src/agent/graph.py`）
2. **节点连接和条件分支**
3. **完整 Agent 的运行**

---

**文档版本**: 1.0  
**创建日期**: 2026-01-13  
**适用项目**: HuahuaChat 阶段三
