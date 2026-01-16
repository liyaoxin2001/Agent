# AgentState 实现指引

## 一、核心概念

### 1.1 什么是 State？

在 LangGraph 中，State 是 Agent 执行过程中的"共享工作空间"：

```
┌──────────────────────────────┐
│      AgentState              │
│  (所有节点共享的数据)         │
├──────────────────────────────┤
│  • question                  │
│  • retrieved_docs            │
│  • answer                    │
│  • messages                  │
│  • ...                       │
└──────────────────────────────┘
    ↓         ↓         ↓
  节点1     节点2     节点3
 (检索)   (生成)   (评估)
```

**执行流程**：
```
1. 初始化 State: {"question": "Python是什么？"}
2. 检索节点: State += {"retrieved_docs": [doc1, doc2]}
3. 生成节点: State += {"answer": "Python是..."}
4. 返回最终 State
```

### 1.2 TypedDict vs 普通 Dict

**为什么使用 TypedDict？**

```python
# ❌ 普通 dict - 没有类型检查
state = {"question": "Python是什么？"}
state["anser"] = "..."  # 拼写错误，运行时才发现！

# ✅ TypedDict - 编辑器会提示错误
class AgentState(TypedDict):
    question: str
    answer: str

state: AgentState = {"question": "Python是什么？"}
state["anser"] = "..."  # IDE 立即提示错误！
```

---

## 二、字段设计

### 2.1 基础版本（推荐从这里开始）

```python
from typing import TypedDict, List, Optional
from langchain.schema import Document

class AgentState(TypedDict):
    """Agent 状态 - 基础版本"""
    
    # 核心字段
    question: str                           # 用户问题
    retrieved_docs: Optional[List[Document]] # 检索到的文档
    answer: Optional[str]                    # 生成的答案
```

**适用场景**：
- ✅ 简单的 RAG 流程：问题 → 检索 → 生成 → 答案
- ✅ 学习 LangGraph 的第一个 Agent
- ✅ 原型开发和快速测试

**示例使用**：
```python
# 初始化
state = AgentState(
    question="Python 的应用领域有哪些？",
    retrieved_docs=None,
    answer=None
)

# 检索节点更新
state["retrieved_docs"] = [doc1, doc2, doc3]

# 生成节点更新
state["answer"] = "Python 广泛应用于..."
```

---

### 2.2 进阶版本（支持多轮对话）

```python
from typing import TypedDict, List, Optional
from langchain.schema import Document, BaseMessage

class AgentState(TypedDict):
    """Agent 状态 - 进阶版本"""
    
    # 当前问题
    question: str
    
    # 检索相关
    retrieved_docs: Optional[List[Document]]
    retrieval_query: Optional[str]  # 实际用于检索的查询（可能经过改写）
    
    # 生成相关
    answer: Optional[str]
    
    # 对话历史（支持多轮对话）
    messages: List[BaseMessage]  # [HumanMessage, AIMessage, ...]
    
    # 执行控制
    step_count: int  # 当前步骤数
    max_steps: int   # 最大步骤数（防止无限循环）
```

**适用场景**：
- ✅ 需要多轮对话
- ✅ 需要查询改写（如：代词替换）
- ✅ 需要控制执行流程

**示例使用**：
```python
# 初始化（第一轮对话）
state = AgentState(
    question="Python是什么？",
    retrieved_docs=None,
    retrieval_query=None,
    answer=None,
    messages=[],
    step_count=0,
    max_steps=5
)

# 第二轮对话（"它"指代 Python）
state["question"] = "它的应用领域有哪些？"
state["retrieval_query"] = "Python 的应用领域有哪些？"  # 查询改写
state["step_count"] = 1
```

---

### 2.3 完整版本（生产级）

```python
from typing import TypedDict, List, Optional, Dict, Any
from langchain.schema import Document, BaseMessage
from datetime import datetime

class AgentState(TypedDict):
    """Agent 状态 - 完整版本"""
    
    # ========== 核心字段 ==========
    question: str                               # 用户原始问题
    answer: Optional[str]                        # 最终答案
    
    # ========== 检索相关 ==========
    retrieved_docs: Optional[List[Document]]     # 检索到的文档
    retrieval_query: Optional[str]               # 改写后的查询
    retrieval_score: Optional[float]             # 检索质量分数
    need_more_context: bool                      # 是否需要更多上下文
    
    # ========== 生成相关 ==========
    intermediate_answer: Optional[str]           # 中间答案（用于多步推理）
    confidence_score: Optional[float]            # 答案置信度
    
    # ========== 对话管理 ==========
    messages: List[BaseMessage]                  # 完整对话历史
    conversation_id: Optional[str]               # 会话 ID
    
    # ========== 执行控制 ==========
    step_count: int                              # 当前步骤
    max_steps: int                               # 最大步骤限制
    current_node: Optional[str]                  # 当前节点名称
    next_action: Optional[str]                   # 下一步动作
    
    # ========== 工具调用（可选）==========
    tool_calls: Optional[List[Dict[str, Any]]]   # 工具调用记录
    tool_results: Optional[List[Any]]            # 工具执行结果
    
    # ========== 元数据 ==========
    metadata: Dict[str, Any]                     # 额外元数据
    start_time: Optional[datetime]               # 开始时间
    error: Optional[str]                         # 错误信息（如果有）
```

**适用场景**：
- ✅ 生产环境部署
- ✅ 需要详细日志和监控
- ✅ 复杂的多步骤 Agent
- ✅ 需要工具调用

---

## 三、字段详解

### 3.1 核心字段

#### `question: str`
- **作用**：存储用户的原始问题
- **示例**：`"Python 的应用领域有哪些？"`
- **注意**：保持原样，不要修改

#### `answer: Optional[str]`
- **作用**：存储最终生成的答案
- **示例**：`"Python 广泛应用于 Web 开发、数据科学..."`
- **为什么 Optional**：初始化时没有答案，由生成节点填充

#### `retrieved_docs: Optional[List[Document]]`
- **作用**：存储从向量库检索到的文档
- **示例**：
  ```python
  [
      Document(page_content="Python 是一种编程语言...", metadata={"source": "python.txt"}),
      Document(page_content="Python 应用于数据科学...", metadata={"source": "ai.txt"})
  ]
  ```
- **为什么 Optional**：初始化时还未检索

---

### 3.2 检索优化字段

#### `retrieval_query: Optional[str]`
- **作用**：经过改写/优化的检索查询
- **应用场景**：
  ```python
  # 场景1: 代词替换
  question = "它的应用领域有哪些？"          # 用户问题
  retrieval_query = "Python 的应用领域有哪些？"  # 改写后

  # 场景2: 查询扩展
  question = "RAG"                           # 用户问题
  retrieval_query = "RAG 检索增强生成 原理"      # 扩展后
  ```

#### `retrieval_score: Optional[float]`
- **作用**：评估检索质量（0.0 - 1.0）
- **用途**：决策是否需要重新检索
  ```python
  if state["retrieval_score"] < 0.5:
      # 检索质量差，需要改写查询重新检索
      return "rewrite_query"
  else:
      return "generate_answer"
  ```

#### `need_more_context: bool`
- **作用**：标记是否需要更多上下文
- **用途**：条件分支决策
  ```python
  def decide_next_step(state):
      if state["need_more_context"]:
          return "retrieve_more"  # 增加检索数量或扩展查询
      else:
          return "generate"
  ```

---

### 3.3 对话管理字段

#### `messages: List[BaseMessage]`
- **作用**：存储完整对话历史
- **结构**：
  ```python
  from langchain.schema import HumanMessage, AIMessage
  
  messages = [
      HumanMessage(content="Python是什么？"),
      AIMessage(content="Python是一种高级编程语言..."),
      HumanMessage(content="它的应用领域有哪些？"),
      AIMessage(content="Python广泛应用于...")
  ]
  ```
- **用途**：
  - 上下文理解（代词消解）
  - 对话式交互
  - 生成时注入历史

---

### 3.4 执行控制字段

#### `step_count: int` 和 `max_steps: int`
- **作用**：防止无限循环
- **示例**：
  ```python
  def should_continue(state):
      if state["step_count"] >= state["max_steps"]:
          return "end"  # 达到最大步数，强制结束
      else:
          return "continue"
  ```

#### `current_node: Optional[str]`
- **作用**：记录当前执行的节点（用于日志和调试）
- **示例**：
  ```python
  def retrieve_node(state):
      state["current_node"] = "retrieve"
      print(f"[{state['current_node']}] 开始检索...")
      # ... 检索逻辑
  ```

---

### 3.5 工具调用字段（可选）

#### `tool_calls: Optional[List[Dict[str, Any]]]`
- **作用**：记录调用了哪些工具
- **示例**：
  ```python
  tool_calls = [
      {
          "tool": "web_search",
          "query": "Python 最新版本",
          "timestamp": "2026-01-13 10:30:00"
      },
      {
          "tool": "calculator",
          "expression": "1024 * 768",
          "timestamp": "2026-01-13 10:30:05"
      }
  ]
  ```

#### `tool_results: Optional[List[Any]]`
- **作用**：存储工具执行结果
- **示例**：
  ```python
  tool_results = [
      {"search_results": ["Python 3.12 发布于 2024 年..."]},
      {"result": 786432}
  ]
  ```

---

## 四、实现建议

### 4.1 渐进式实现

**第1步：基础版本**（今天实现）
```python
class AgentState(TypedDict):
    question: str
    retrieved_docs: Optional[List[Document]]
    answer: Optional[str]
```

**第2步：添加对话历史**（明天）
```python
class AgentState(TypedDict):
    question: str
    retrieved_docs: Optional[List[Document]]
    answer: Optional[str]
    messages: List[BaseMessage]  # 新增
```

**第3步：添加执行控制**（后天）
```python
class AgentState(TypedDict):
    # ... 之前的字段
    step_count: int      # 新增
    max_steps: int       # 新增
```

### 4.2 代码组织

**推荐结构**：
```python
# src/agent/state.py

from typing import TypedDict, List, Optional, Dict, Any
from langchain.schema import Document, BaseMessage


class AgentState(TypedDict):
    """
    LangGraph Agent 状态定义
    
    State 是 Agent 执行过程中的共享工作空间，
    记录问题、检索结果、答案等信息。
    """
    
    # ========== 核心字段 ==========
    question: str
    """用户提出的问题"""
    
    retrieved_docs: Optional[List[Document]]
    """从向量库检索到的相关文档"""
    
    answer: Optional[str]
    """LLM 生成的最终答案"""
    
    # ========== 对话历史 ==========
    messages: List[BaseMessage]
    """完整的对话历史记录"""
    
    # ========== 执行控制 ==========
    step_count: int
    """当前执行的步骤数（从 0 开始）"""
    
    max_steps: int
    """允许的最大步骤数（防止无限循环）"""


# 辅助函数：创建初始状态
def create_initial_state(
    question: str,
    max_steps: int = 5
) -> AgentState:
    """
    创建初始 Agent 状态
    
    Args:
        question: 用户问题
        max_steps: 最大执行步数
        
    Returns:
        初始化的 AgentState
    """
    return AgentState(
        question=question,
        retrieved_docs=None,
        answer=None,
        messages=[],
        step_count=0,
        max_steps=max_steps
    )
```

---

## 五、使用示例

### 5.1 初始化 State

```python
from src.agent.state import AgentState, create_initial_state

# 方法1: 直接创建
state = AgentState(
    question="Python 的应用领域有哪些？",
    retrieved_docs=None,
    answer=None,
    messages=[],
    step_count=0,
    max_steps=5
)

# 方法2: 使用辅助函数（推荐）
state = create_initial_state(
    question="Python 的应用领域有哪些？",
    max_steps=5
)
```

### 5.2 在节点中更新 State

```python
def retrieve_node(state: AgentState) -> AgentState:
    """检索节点"""
    print(f"📖 正在检索问题: {state['question']}")
    
    # 执行检索
    docs = vectorstore.similarity_search(state["question"], k=4)
    
    # 更新 State
    state["retrieved_docs"] = docs
    state["step_count"] += 1
    
    print(f"✅ 检索到 {len(docs)} 个相关文档")
    return state


def generate_node(state: AgentState) -> AgentState:
    """生成节点"""
    print(f"🤖 正在生成答案...")
    
    # 组装上下文
    context = "\n\n".join([
        doc.page_content for doc in state["retrieved_docs"]
    ])
    
    # 组装 Prompt
    prompt = f"""基于以下上下文回答问题：

上下文：
{context}

问题：{state['question']}

答案："""
    
    # 生成答案
    answer = llm.generate(prompt)
    
    # 更新 State
    state["answer"] = answer
    state["step_count"] += 1
    
    print(f"✅ 答案生成完成")
    return state
```

### 5.3 条件判断

```python
def should_continue(state: AgentState) -> str:
    """决定是否继续执行"""
    
    # 检查步骤数
    if state["step_count"] >= state["max_steps"]:
        print("⚠️ 达到最大步骤数，结束执行")
        return "end"
    
    # 检查是否已生成答案
    if state["answer"] is not None:
        print("✅ 答案已生成，结束执行")
        return "end"
    
    # 继续执行
    return "continue"
```

---

## 六、常见问题

### Q1: 为什么很多字段是 `Optional`？

**A**: 因为这些字段在初始化时还没有值，会在后续节点中填充。

```python
# 初始化时
state = {
    "question": "Python是什么？",
    "retrieved_docs": None,      # 还未检索
    "answer": None               # 还未生成
}

# 检索节点后
state["retrieved_docs"] = [doc1, doc2]  # 填充了

# 生成节点后
state["answer"] = "Python是..."  # 填充了
```

### Q2: State 会不会越来越大，占用太多内存？

**A**: 不会。State 只在单次执行中存在，执行结束后就会释放。

```python
# 每次对话都是新的 State
state1 = create_initial_state("问题1")  # 独立的 State
result1 = graph.invoke(state1)
# state1 执行完毕，内存释放

state2 = create_initial_state("问题2")  # 新的 State
result2 = graph.invoke(state2)
```

### Q3: 如何在 State 中添加自定义字段？

**A**: 直接在 `AgentState` 中定义即可。

```python
class AgentState(TypedDict):
    # 标准字段
    question: str
    answer: Optional[str]
    
    # 自定义字段
    user_id: str                    # 用户ID
    knowledge_base: str             # 使用的知识库
    temperature: float              # LLM 温度参数
    debug_info: Dict[str, Any]      # 调试信息
```

### Q4: 节点必须返回完整的 State 吗？

**A**: 可以只返回更新的字段（部分更新）。

```python
# 方式1: 返回完整 State
def node1(state: AgentState) -> AgentState:
    state["retrieved_docs"] = docs
    return state  # 返回整个 state

# 方式2: 只返回更新的字段（LangGraph 会自动合并）
def node2(state: AgentState) -> dict:
    return {"answer": "Python是..."}  # 只返回更新部分
```

---

## 七、下一步

完成 State 定义后，你将学习：

1. **节点实现**（`src/agent/nodes.py`）
   - 如何编写 retrieve_node
   - 如何编写 generate_node
   - 如何编写 decide_node

2. **图构建**（`src/agent/graph.py`）
   - 如何连接节点
   - 如何实现条件分支
   - 如何设置起点和终点

3. **测试运行**
   - 如何执行 Agent
   - 如何查看中间状态
   - 如何调试问题

---

## 八、学习资源

- [LangGraph 官方文档 - State](https://langchain-ai.github.io/langgraph/concepts/low_level/#state)
- [LangGraph 教程 - 构建第一个 Agent](https://langchain-ai.github.io/langgraph/tutorials/introduction/)
- [TypedDict 文档](https://docs.python.org/3/library/typing.html#typing.TypedDict)

---

**记住**：从简单开始，逐步添加复杂功能！🚀
