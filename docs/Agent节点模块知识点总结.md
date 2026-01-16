# Agent 节点模块知识点总结

## 📋 文档概述

本文档详细总结 LangGraph Agent 节点（Node）模块的核心知识点，包括概念、设计模式、实现方法、最佳实践和常见问题。

**适用对象**：学习 LangGraph Agent 开发的开发者

**前置知识**：建议先阅读 `AgentState模块知识点总结.md`

**知识点层级**：从基础概念到高级应用，逐层递进

---

## 一、核心概念

### 1.1 什么是节点（Node）？

#### 定义

**节点（Node）** 是 LangGraph Agent 中的一个**处理单元**，负责执行特定的业务逻辑。

**本质**：节点是一个**纯函数**
```
输入：AgentState（状态）
处理：执行业务逻辑
输出：更新后的 AgentState
```

**函数签名**：
```python
def node_name(state: AgentState) -> AgentState:
    """节点函数"""
    # 1. 读取 State 中的数据
    data = state["some_field"]
    
    # 2. 执行业务逻辑
    result = process_data(data)
    
    # 3. 更新 State
    state["output_field"] = result
    
    # 4. 返回更新后的 State
    return state
```

---

#### 类比理解

**1. 流水线工位**

```
产品（State）在流水线上流转：

原料 → [工位1: 切割] → 半成品A → [工位2: 组装] → 半成品B → [工位3: 包装] → 成品

State  →  节点1      → State'    →  节点2      → State''   →  节点3    → 最终State
```

每个工位（节点）：
- 接收产品（State）
- 进行加工（执行逻辑）
- 传给下一个工位（返回更新后的 State）

**2. 数据管道**

```
原始数据 → [清洗] → 干净数据 → [转换] → 结构化数据 → [分析] → 分析结果

State   → 节点1  → State'   → 节点2  → State''     → 节点3  → 最终State
```

**3. 任务分解**

```
完成一篇文章：

[收集资料] → [整理大纲] → [撰写初稿] → [修改润色] → [最终定稿]
   节点1        节点2         节点3         节点4        完成

每个节点负责一个明确的子任务
```

---

### 1.2 节点的核心特性

#### 特性1：纯函数特性

**什么是纯函数？**
- 相同输入总是产生相同输出
- 无副作用（不修改外部状态）
- 只通过 State 与外界交互

```python
# ✅ 纯函数节点 - 推荐
def retrieve_node(state: AgentState) -> AgentState:
    """只通过 State 交互"""
    docs = vectorstore.search(state["question"])
    state["retrieved_docs"] = docs
    return state

# ❌ 非纯函数 - 有副作用
global_result = None  # 全局变量

def bad_node(state: AgentState) -> AgentState:
    """修改了全局状态"""
    global global_result
    global_result = process(state["question"])  # 副作用！
    state["result"] = global_result
    return state
```

**为什么需要纯函数？**
1. **可测试性**：相同输入 → 相同输出，容易测试
2. **可预测性**：无副作用，行为可预测
3. **可组合性**：纯函数可以任意组合
4. **可调试性**：问题容易定位

---

#### 特性2：职责单一

**单一职责原则（SRP）**：一个节点只做一件事

```python
# ✅ 职责单一 - 推荐
def retrieve_node(state):
    """只做检索"""
    docs = vectorstore.search(state["question"])
    state["retrieved_docs"] = docs
    return state

def generate_node(state):
    """只做生成"""
    answer = llm.generate(state["retrieved_docs"])
    state["answer"] = answer
    return state

# ❌ 职责混乱 - 不推荐
def retrieve_and_generate_node(state):
    """既检索又生成，职责不清"""
    docs = vectorstore.search(state["question"])
    answer = llm.generate(docs)
    state["answer"] = answer
    return state
```

**为什么要职责单一？**

1. **易于理解**：一个节点做一件事，容易理解
2. **易于测试**：单一功能，测试简单
3. **易于复用**：职责单一的节点更容易在不同场景复用
4. **易于维护**：修改一个功能只需改一个节点

**如何判断职责是否单一？**
- 节点名称是否能用一个动词描述？（retrieve、generate、evaluate）
- 节点是否只更新 State 中的一个或少数几个字段？
- 节点的代码是否可以在 30 行内完成？

---

#### 特性3：可组合性

**节点像积木一样可以组合**

```python
# 节点1: 检索
def retrieve_node(state):
    docs = vectorstore.search(state["question"])
    state["retrieved_docs"] = docs
    return state

# 节点2: 重排序
def rerank_node(state):
    docs = rerank(state["retrieved_docs"])
    state["retrieved_docs"] = docs  # 覆盖原有文档
    return state

# 节点3: 生成
def generate_node(state):
    answer = llm.generate(state["retrieved_docs"])
    state["answer"] = answer
    return state

# 组合方式1: 简单流程
state = retrieve_node(state)
state = generate_node(state)

# 组合方式2: 复杂流程
state = retrieve_node(state)
state = rerank_node(state)  # 插入重排序
state = generate_node(state)

# 组合方式3: 循环流程
state = retrieve_node(state)
if state["retrieval_score"] < 0.6:
    state = retrieve_node(state)  # 重新检索
state = generate_node(state)
```

**组合的优势**：
- ✅ 灵活性：可以任意组合节点
- ✅ 可扩展性：添加新节点不影响现有节点
- ✅ 可维护性：每个节点独立维护

---

#### 特性4：无状态性

**节点本身不保存状态，状态都在 State 中**

```python
# ✅ 无状态节点 - 推荐
def retrieve_node(state: AgentState) -> AgentState:
    """不保存内部状态"""
    docs = vectorstore.search(state["question"])
    state["retrieved_docs"] = docs  # 状态保存在 State 中
    return state

# ❌ 有状态节点 - 不推荐
class StatefulNode:
    def __init__(self):
        self.cached_docs = []  # 内部状态
    
    def retrieve(self, state):
        """保存了内部状态，难以测试和调试"""
        if not self.cached_docs:
            self.cached_docs = vectorstore.search(state["question"])
        state["retrieved_docs"] = self.cached_docs
        return state
```

**为什么要无状态？**
1. **并发安全**：无状态节点可以并发执行
2. **易于测试**：每次调用都是独立的
3. **易于调试**：所有状态都在 State 中，容易追踪

---

### 1.3 节点的分类

根据功能，节点可以分为以下几类：

#### 分类1：数据处理节点

**作用**：处理和转换数据

**示例**：
- `retrieve_node`：检索文档
- `parse_node`：解析文档
- `transform_node`：数据转换

```python
def retrieve_node(state):
    """检索节点 - 从向量库获取文档"""
    docs = vectorstore.search(state["question"], k=4)
    state["retrieved_docs"] = docs
    return state

def parse_node(state):
    """解析节点 - 提取文档中的关键信息"""
    docs = state["retrieved_docs"]
    key_points = extract_key_points(docs)
    state["key_points"] = key_points
    return state
```

---

#### 分类2：生成节点

**作用**：调用 LLM 生成内容

**示例**：
- `generate_node`：生成答案
- `summarize_node`：生成摘要
- `translate_node`：翻译文本

```python
def generate_node(state):
    """生成节点 - 基于文档生成答案"""
    context = format_context(state["retrieved_docs"])
    prompt = f"基于上下文回答：{context}\n问题：{state['question']}"
    answer = llm.generate(prompt)
    state["answer"] = answer
    return state

def summarize_node(state):
    """摘要节点 - 生成文档摘要"""
    docs = state["retrieved_docs"]
    summary = llm.generate(f"请总结以下内容：{docs}")
    state["summary"] = summary
    return state
```

---

#### 分类3：决策节点

**作用**：决定执行流程

**特殊性**：返回字符串（节点名），而不是 State

```python
def decide_node(state):
    """决策节点 - 决定下一步操作"""
    
    # 决策逻辑
    if state.get("answer"):
        return "end"  # 已有答案，结束
    
    if state.get("retrieved_docs") is None:
        return "retrieve"  # 还未检索
    
    if state.get("retrieval_score") < 0.6:
        return "rewrite_query"  # 质量差，重写查询
    
    return "generate"  # 生成答案
```

**关键点**：
- 返回类型是 `str`，不是 `AgentState`
- 返回值是下一个节点的名称
- 用于实现条件分支

---

#### 分类4：评估节点

**作用**：评估质量和性能

**示例**：
- `evaluate_retrieval_node`：评估检索质量
- `evaluate_answer_node`：评估答案质量
- `score_node`：计算分数

```python
def evaluate_retrieval_node(state):
    """评估检索质量"""
    docs = state["retrieved_docs"]
    question = state["question"]
    
    # 计算相关性分数
    relevance_scores = [
        calculate_relevance(doc, question)
        for doc in docs
    ]
    
    avg_score = sum(relevance_scores) / len(relevance_scores)
    state["retrieval_score"] = avg_score
    
    # 判断是否需要更多上下文
    state["need_more_context"] = avg_score < 0.6
    
    return state

def evaluate_answer_node(state):
    """评估答案质量"""
    answer = state["answer"]
    question = state["question"]
    
    # 多维度评估
    length_score = min(len(answer) / 100, 1.0)
    keyword_score = calculate_keyword_match(question, answer)
    
    state["confidence_score"] = (length_score + keyword_score) / 2
    
    return state
```

---

#### 分类5：工具调用节点

**作用**：调用外部工具或API

**示例**：
- `web_search_node`：网络搜索
- `calculator_node`：计算器
- `database_query_node`：数据库查询

```python
def web_search_node(state):
    """网络搜索节点"""
    query = state["question"]
    
    # 调用搜索 API
    search_results = search_api.search(query)
    
    # 记录工具调用
    state.setdefault("tool_calls", []).append({
        "tool": "web_search",
        "query": query,
        "timestamp": datetime.now()
    })
    
    # 保存结果
    state["search_results"] = search_results
    
    return state

def calculator_node(state):
    """计算器节点"""
    expression = extract_math_expression(state["question"])
    
    if expression:
        result = eval(expression)  # 实际应用中要安全地执行
        state["calculation_result"] = result
    
    return state
```

---

#### 分类6：辅助节点

**作用**：辅助功能，如日志、格式化等

**示例**：
- `log_node`：记录日志
- `format_node`：格式化输出
- `validate_node`：验证数据

```python
def log_node(state):
    """日志节点 - 记录执行信息"""
    state.setdefault("processing_log", []).append({
        "step": state.get("step_count", 0),
        "timestamp": datetime.now(),
        "status": "processing"
    })
    return state

def format_node(state):
    """格式化节点 - 格式化输出"""
    answer = state.get("answer", "")
    
    # 添加引用来源
    docs = state.get("retrieved_docs", [])
    sources = [doc.metadata.get("source", "未知") for doc in docs]
    
    formatted_answer = f"{answer}\n\n参考来源：\n"
    formatted_answer += "\n".join(f"- {s}" for s in set(sources))
    
    state["formatted_answer"] = formatted_answer
    
    return state

def validate_node(state):
    """验证节点 - 验证数据完整性"""
    errors = []
    
    if not state.get("question"):
        errors.append("缺少问题")
    
    if not state.get("retrieved_docs"):
        errors.append("缺少检索文档")
    
    if errors:
        state["validation_errors"] = errors
        state["is_valid"] = False
    else:
        state["is_valid"] = True
    
    return state
```

---

## 二、节点设计模式

### 2.1 工厂模式（推荐）

#### 为什么需要工厂模式？

**问题**：节点需要访问外部资源（LLM、VectorStore等），如何优雅地注入依赖？

```python
# ❌ 问题1：全局变量
vectorstore = None  # 全局变量

def retrieve_node(state):
    global vectorstore  # 依赖全局变量
    docs = vectorstore.search(state["question"])
    return state

# 缺点：
# - 难以测试（无法替换为 mock）
# - 难以配置（无法传入不同的 vectorstore）
# - 全局状态，线程不安全

# ❌ 问题2：硬编码
def retrieve_node(state):
    # 硬编码创建对象
    vectorstore = FAISSVectorStore(
        embedding=OpenAIEmbedding(),
        persist_directory="./data"
    )
    docs = vectorstore.search(state["question"])
    return state

# 缺点：
# - 每次调用都创建新对象（性能差）
# - 无法复用
# - 难以测试
```

---

#### 工厂模式解决方案

**核心思想**：使用工厂函数创建节点，在创建时注入依赖

```python
def create_retrieve_node(vectorstore: BaseVectorStore, k: int = 4):
    """
    工厂函数：创建检索节点
    
    Args:
        vectorstore: 向量存储实例（依赖注入）
        k: 检索数量（参数配置）
        
    Returns:
        检索节点函数
    """
    def retrieve_node(state: AgentState) -> AgentState:
        """实际的节点函数"""
        # 使用闭包访问 vectorstore 和 k
        docs = vectorstore.similarity_search(
            state["question"],
            k=k
        )
        state["retrieved_docs"] = docs
        return state
    
    return retrieve_node  # 返回节点函数
```

**使用方式**：

```python
# 创建时注入依赖
my_vectorstore = FAISSVectorStore(...)
retrieve_node = create_retrieve_node(my_vectorstore, k=4)

# 使用节点
state = {"question": "Python是什么？"}
state = retrieve_node(state)

# 测试时注入 mock
mock_vectorstore = MockVectorStore()
test_retrieve_node = create_retrieve_node(mock_vectorstore, k=2)
```

---

#### 工厂模式的优势

**1. 依赖注入**

```python
# 生产环境
prod_vectorstore = FAISSVectorStore(...)
prod_retrieve_node = create_retrieve_node(prod_vectorstore)

# 测试环境
test_vectorstore = MockVectorStore()
test_retrieve_node = create_retrieve_node(test_vectorstore)

# 本地开发
dev_vectorstore = ChromaVectorStore(...)
dev_retrieve_node = create_retrieve_node(dev_vectorstore)
```

**2. 参数配置**

```python
# 不同的检索数量
retrieve_few = create_retrieve_node(vectorstore, k=2)
retrieve_many = create_retrieve_node(vectorstore, k=10)

# 不同的 LLM 配置
generate_fast = create_generate_node(
    llm=OpenAILLM(model="gpt-3.5-turbo", temperature=0.3)
)
generate_creative = create_generate_node(
    llm=OpenAILLM(model="gpt-4", temperature=0.9)
)
```

**3. 易于测试**

```python
# 测试检索节点
def test_retrieve_node():
    # 创建 mock
    mock_vectorstore = MockVectorStore()
    mock_vectorstore.set_return_docs([doc1, doc2])
    
    # 创建节点
    retrieve_node = create_retrieve_node(mock_vectorstore, k=2)
    
    # 测试
    state = {"question": "test"}
    result = retrieve_node(state)
    
    # 验证
    assert len(result["retrieved_docs"]) == 2
```

---

#### 工厂模式最佳实践

**1. 参数命名清晰**

```python
# ✅ 好的命名
def create_retrieve_node(
    vectorstore: BaseVectorStore,  # 明确类型
    k: int = 4,                     # 默认值
    score_threshold: float = 0.7    # 描述性命名
):
    pass

# ❌ 不好的命名
def create_node(vs, n=4, t=0.7):  # 太简短
    pass
```

**2. 提供合理的默认值**

```python
def create_generate_node(
    llm: BaseLLM,
    prompt_template: Optional[str] = None,  # None 表示使用默认模板
    max_tokens: int = 500,                  # 合理的默认值
    temperature: float = 0.7
):
    # 使用默认模板
    template = prompt_template or get_default_template()
    
    def generate_node(state):
        # ...
        return state
    
    return generate_node
```

**3. 文档字符串完整**

```python
def create_retrieve_node(
    vectorstore: BaseVectorStore,
    k: int = 4,
    filter_func: Optional[Callable] = None
):
    """
    创建检索节点
    
    Args:
        vectorstore: 向量存储实例
        k: 检索的文档数量（默认 4）
        filter_func: 可选的文档过滤函数
        
    Returns:
        检索节点函数
        
    示例:
        >>> vectorstore = FAISSVectorStore(...)
        >>> retrieve_node = create_retrieve_node(vectorstore, k=4)
        >>> state = retrieve_node(state)
    """
    def retrieve_node(state):
        # ...
        return state
    
    return retrieve_node
```

---

### 2.2 类方法模式（可选）

#### 使用场景

当多个节点需要共享配置或状态时，可以使用类来组织

```python
class AgentNodes:
    """Agent 节点集合类"""
    
    def __init__(
        self,
        llm: BaseLLM,
        vectorstore: BaseVectorStore,
        config: Dict[str, Any]
    ):
        """初始化节点集合"""
        self.llm = llm
        self.vectorstore = vectorstore
        self.config = config
    
    def retrieve(self, state: AgentState) -> AgentState:
        """检索节点"""
        k = self.config.get("retrieve_k", 4)
        docs = self.vectorstore.similarity_search(
            state["question"],
            k=k
        )
        state["retrieved_docs"] = docs
        return state
    
    def generate(self, state: AgentState) -> AgentState:
        """生成节点"""
        temperature = self.config.get("temperature", 0.7)
        
        # 使用配置的温度参数
        answer = self.llm.generate(
            prompt=self._build_prompt(state),
            temperature=temperature
        )
        state["answer"] = answer
        return state
    
    def _build_prompt(self, state: AgentState) -> str:
        """辅助方法：构建 prompt"""
        context = "\n".join([
            doc.page_content
            for doc in state["retrieved_docs"]
        ])
        return f"上下文：{context}\n问题：{state['question']}"
```

**使用方式**：

```python
# 创建节点集合
nodes = AgentNodes(
    llm=OpenAILLM(...),
    vectorstore=FAISSVectorStore(...),
    config={
        "retrieve_k": 4,
        "temperature": 0.7
    }
)

# 使用节点
state = nodes.retrieve(state)
state = nodes.generate(state)
```

---

#### 类方法模式的优势

**1. 共享配置**

```python
class AgentNodes:
    def __init__(self, config):
        self.config = config  # 所有节点共享配置
    
    def retrieve(self, state):
        k = self.config["retrieve_k"]  # 使用共享配置
        # ...
        return state
    
    def generate(self, state):
        temp = self.config["temperature"]  # 使用共享配置
        # ...
        return state
```

**2. 代码组织清晰**

```python
class AgentNodes:
    """所有节点都在一个类中，组织清晰"""
    
    # 数据处理节点
    def retrieve(self, state): ...
    def rerank(self, state): ...
    
    # 生成节点
    def generate(self, state): ...
    def summarize(self, state): ...
    
    # 辅助方法
    def _build_prompt(self, state): ...
    def _calculate_score(self, docs): ...
```

**3. 便于继承扩展**

```python
class BasicAgentNodes:
    """基础节点"""
    def retrieve(self, state): ...
    def generate(self, state): ...

class AdvancedAgentNodes(BasicAgentNodes):
    """扩展节点"""
    def retrieve(self, state):
        """重写检索逻辑"""
        # 先调用父类方法
        state = super().retrieve(state)
        
        # 添加额外处理
        state = self.rerank(state)
        return state
    
    def rerank(self, state):
        """新增重排序节点"""
        # ...
        return state
```

---

#### 何时使用类方法模式？

**✅ 适合使用的场景**：
- 多个节点需要共享配置
- 节点间有辅助方法需要复用
- 需要继承和扩展节点功能
- 项目结构较复杂，需要清晰的组织

**❌ 不适合使用的场景**：
- 简单的 Agent（只有 2-3 个节点）
- 节点完全独立，无共享需求
- 追求函数式编程风格

**推荐**：
- 小型项目：使用工厂模式
- 大型项目：使用类方法模式

---

### 2.3 装饰器模式（高级）

#### 使用场景

为节点添加通用功能（日志、错误处理、性能监控等）

```python
import functools
from typing import Callable

def with_error_handling(node_func: Callable) -> Callable:
    """错误处理装饰器"""
    @functools.wraps(node_func)
    def wrapper(state: AgentState) -> AgentState:
        try:
            return node_func(state)
        except Exception as e:
            # 记录错误到 State
            state["error"] = f"{node_func.__name__} 失败: {str(e)}"
            return state
    
    return wrapper

def with_logging(node_name: str):
    """日志装饰器（参数化）"""
    def decorator(node_func: Callable) -> Callable:
        @functools.wraps(node_func)
        def wrapper(state: AgentState) -> AgentState:
            # 记录开始
            print(f"[{node_name}] 开始执行...")
            state.setdefault("processing_log", []).append(
                f"[{node_name}] 开始"
            )
            
            # 执行节点
            result = node_func(state)
            
            # 记录结束
            print(f"[{node_name}] 执行完成")
            state["processing_log"].append(f"[{node_name}] 完成")
            
            return result
        
        return wrapper
    
    return decorator

def with_timing(node_func: Callable) -> Callable:
    """性能计时装饰器"""
    @functools.wraps(node_func)
    def wrapper(state: AgentState) -> AgentState:
        import time
        
        start_time = time.time()
        result = node_func(state)
        elapsed = time.time() - start_time
        
        # 记录耗时
        state.setdefault("node_timings", {})[node_func.__name__] = elapsed
        
        return result
    
    return wrapper
```

**使用方式**：

```python
# 单个装饰器
@with_error_handling
def retrieve_node(state):
    docs = vectorstore.search(state["question"])
    state["retrieved_docs"] = docs
    return state

# 多个装饰器（从下到上执行）
@with_logging("retrieve")
@with_timing
@with_error_handling
def retrieve_node(state):
    docs = vectorstore.search(state["question"])
    state["retrieved_docs"] = docs
    return state

# 执行顺序：
# 1. with_logging 开始日志
# 2. with_timing 开始计时
# 3. with_error_handling 错误处理
# 4. 实际的 retrieve_node 执行
# 5. with_error_handling 完成
# 6. with_timing 记录时间
# 7. with_logging 结束日志
```

---

#### 装饰器模式的优势

**1. 关注点分离**

```python
# 业务逻辑与横切关注点（日志、错误处理）分离

# 纯粹的业务逻辑
def retrieve_node(state):
    docs = vectorstore.search(state["question"])
    state["retrieved_docs"] = docs
    return state

# 通过装饰器添加横切关注点
@with_logging("retrieve")
@with_error_handling
@with_timing
def retrieve_node(state):
    # 只关注业务逻辑
    docs = vectorstore.search(state["question"])
    state["retrieved_docs"] = docs
    return state
```

**2. 代码复用**

```python
# 装饰器可以应用于任意节点

@with_error_handling
def retrieve_node(state): ...

@with_error_handling
def generate_node(state): ...

@with_error_handling
def evaluate_node(state): ...

# 所有节点都有了错误处理能力
```

**3. 灵活组合**

```python
# 开发环境：详细日志 + 性能监控
@with_logging("retrieve")
@with_timing
def retrieve_node(state): ...

# 生产环境：只要错误处理
@with_error_handling
def retrieve_node(state): ...

# 测试环境：什么都不加
def retrieve_node(state): ...
```

---

#### 常用装饰器示例

**1. 重试装饰器**

```python
def with_retry(max_retries: int = 3, delay: float = 1.0):
    """重试装饰器"""
    def decorator(node_func):
        @functools.wraps(node_func)
        def wrapper(state):
            for attempt in range(max_retries):
                try:
                    return node_func(state)
                except Exception as e:
                    if attempt == max_retries - 1:
                        # 最后一次尝试失败，记录错误
                        state["error"] = f"重试{max_retries}次后失败: {str(e)}"
                        return state
                    
                    # 等待后重试
                    time.sleep(delay)
        
        return wrapper
    
    return decorator

# 使用
@with_retry(max_retries=3, delay=1.0)
def retrieve_node(state):
    # 如果检索失败，会自动重试3次
    docs = vectorstore.search(state["question"])
    state["retrieved_docs"] = docs
    return state
```

**2. 缓存装饰器**

```python
def with_cache(cache_key_func: Callable):
    """缓存装饰器"""
    cache = {}
    
    def decorator(node_func):
        @functools.wraps(node_func)
        def wrapper(state):
            # 生成缓存键
            key = cache_key_func(state)
            
            # 检查缓存
            if key in cache:
                print(f"[Cache] 命中缓存: {key}")
                state.update(cache[key])
                return state
            
            # 执行节点
            result = node_func(state)
            
            # 保存到缓存
            cache[key] = {
                k: v for k, v in result.items()
                if k not in state  # 只缓存新增的字段
            }
            
            return result
        
        return wrapper
    
    return decorator

# 使用
@with_cache(lambda state: state["question"])
def retrieve_node(state):
    """相同问题会使用缓存的检索结果"""
    docs = vectorstore.search(state["question"])
    state["retrieved_docs"] = docs
    return state
```

**3. 验证装饰器**

```python
def with_validation(required_fields: List[str]):
    """验证装饰器"""
    def decorator(node_func):
        @functools.wraps(node_func)
        def wrapper(state):
            # 验证必需字段
            missing = [f for f in required_fields if f not in state]
            
            if missing:
                state["error"] = f"缺少必需字段: {', '.join(missing)}"
                return state
            
            # 验证通过，执行节点
            return node_func(state)
        
        return wrapper
    
    return decorator

# 使用
@with_validation(["retrieved_docs", "question"])
def generate_node(state):
    """确保有检索文档和问题才生成"""
    # ...
    return state
```

---

## 三、核心节点实现详解

### 3.1 检索节点（Retrieve Node）

#### 作用和职责

**核心职责**：从向量数据库中检索与问题相关的文档

**输入**：
- `state["question"]`：用户问题
- `state["retrieval_query"]`（可选）：改写后的查询

**输出**：
- `state["retrieved_docs"]`：检索到的文档列表
- `state["retrieval_score"]`：检索质量分数
- `state["need_more_context"]`：是否需要更多上下文

---

#### 完整实现

```python
def create_retrieve_node(
    vectorstore: BaseVectorStore,
    k: int = 4,
    score_threshold: Optional[float] = None
):
    """
    创建检索节点
    
    Args:
        vectorstore: 向量存储实例
        k: 检索的文档数量
        score_threshold: 分数阈值（低于此分数的文档会被过滤）
        
    Returns:
        检索节点函数
    """
    def retrieve_node(state: AgentState) -> AgentState:
        """检索节点：从向量库检索相关文档"""
        
        try:
            # ========== 步骤1: 获取查询文本 ==========
            # 优先使用改写后的查询，如果没有则使用原始问题
            query = state.get("retrieval_query") or state["question"]
            
            # ========== 步骤2: 执行检索 ==========
            docs = vectorstore.similarity_search(query, k=k)
            
            # ========== 步骤3: 过滤低分文档（如果设置了阈值）==========
            if score_threshold is not None:
                docs = [
                    doc for doc in docs
                    if doc.metadata.get("score", 1.0) >= score_threshold
                ]
            
            # ========== 步骤4: 计算检索质量分数 ==========
            if docs:
                # 方法1: 如果文档有分数元数据
                if hasattr(docs[0], 'metadata') and 'score' in docs[0].metadata:
                    scores = [doc.metadata.get('score', 0) for doc in docs]
                    retrieval_score = sum(scores) / len(scores)
                else:
                    # 方法2: 基于检索到的文档数量
                    retrieval_score = min(len(docs) / k, 1.0)
            else:
                retrieval_score = 0.0
            
            # ========== 步骤5: 更新 State ==========
            state["retrieved_docs"] = docs
            state["retrieval_score"] = retrieval_score
            state["current_node"] = "retrieve"
            state["step_count"] = state.get("step_count", 0) + 1
            
            # ========== 步骤6: 判断是否需要更多上下文 ==========
            # 条件1: 检索质量低
            # 条件2: 文档数量不足
            state["need_more_context"] = (
                retrieval_score < 0.6 or len(docs) < k // 2
            )
            
            return state
            
        except Exception as e:
            # 错误处理：记录错误但不中断流程
            state["error"] = f"检索失败: {str(e)}"
            state["retrieved_docs"] = []
            state["retrieval_score"] = 0.0
            state["need_more_context"] = True
            return state
    
    return retrieve_node
```

---

#### 关键点解析

**1. 查询选择逻辑**

```python
# 为什么这样写？
query = state.get("retrieval_query") or state["question"]

# 场景1: 第一次检索，没有改写查询
state = {
    "question": "Python是什么？",
    "retrieval_query": None
}
query = None or "Python是什么？"  # 使用原始问题

# 场景2: 查询已被改写（如代词消解）
state = {
    "question": "它的应用领域有哪些？",
    "retrieval_query": "Python的应用领域有哪些？"
}
query = "Python的应用领域有哪些？"  # 使用改写后的查询
```

**2. 分数过滤**

```python
if score_threshold is not None:
    docs = [
        doc for doc in docs
        if doc.metadata.get("score", 1.0) >= score_threshold
    ]

# 用途：过滤相关性太低的文档
# 示例：
# - score_threshold = 0.7
# - 原始检索结果：[doc1(0.9), doc2(0.8), doc3(0.5), doc4(0.4)]
# - 过滤后：[doc1(0.9), doc2(0.8)]
```

**3. 质量分数计算**

```python
# 方法1: 基于文档自带分数（推荐）
if 'score' in docs[0].metadata:
    scores = [doc.metadata.get('score', 0) for doc in docs]
    retrieval_score = sum(scores) / len(scores)
    # 平均分数作为整体质量

# 方法2: 基于数量（简单有效）
else:
    retrieval_score = min(len(docs) / k, 1.0)
    # 如果期望4个文档，检索到4个 → 分数1.0
    # 如果期望4个文档，检索到2个 → 分数0.5
```

**4. 需要更多上下文的判断**

```python
state["need_more_context"] = (
    retrieval_score < 0.6 or  # 质量差
    len(docs) < k // 2         # 数量不足（少于期望的一半）
)

# 用途：触发后续操作
# - 如果 need_more_context = True
#   → 重写查询
#   → 增加检索数量
#   → 扩展查询词
```

---

#### 变体和扩展

**变体1: 带重排序的检索节点**

```python
def create_retrieve_with_rerank_node(vectorstore, reranker, k=4):
    """带重排序的检索节点"""
    def retrieve_node(state):
        # 步骤1: 初次检索（多检索一些）
        docs = vectorstore.similarity_search(
            state["question"],
            k=k*2  # 检索2倍数量
        )
        
        # 步骤2: 重排序
        reranked_docs = reranker.rerank(
            query=state["question"],
            documents=docs
        )
        
        # 步骤3: 取前 k 个
        top_docs = reranked_docs[:k]
        
        state["retrieved_docs"] = top_docs
        return state
    
    return retrieve_node
```

**变体2: 多知识库检索节点**

```python
def create_multi_kb_retrieve_node(vectorstores_dict, k_per_kb=2):
    """从多个知识库检索"""
    def retrieve_node(state):
        all_docs = []
        
        # 从每个知识库检索
        for kb_name, vectorstore in vectorstores_dict.items():
            docs = vectorstore.similarity_search(
                state["question"],
                k=k_per_kb
            )
            
            # 标记来源知识库
            for doc in docs:
                doc.metadata["knowledge_base"] = kb_name
            
            all_docs.extend(docs)
        
        # 合并并按分数排序
        all_docs.sort(
            key=lambda x: x.metadata.get("score", 0),
            reverse=True
        )
        
        state["retrieved_docs"] = all_docs
        return state
    
    return retrieve_node
```

**变体3: 带过滤的检索节点**

```python
def create_filtered_retrieve_node(vectorstore, filter_func, k=4):
    """带自定义过滤的检索节点"""
    def retrieve_node(state):
        # 检索更多文档
        docs = vectorstore.similarity_search(
            state["question"],
            k=k*3
        )
        
        # 应用过滤函数
        filtered_docs = [doc for doc in docs if filter_func(doc, state)]
        
        # 取前 k 个
        state["retrieved_docs"] = filtered_docs[:k]
        return state
    
    return retrieve_node

# 使用示例
def only_recent_docs(doc, state):
    """只保留最近的文档"""
    from datetime import datetime, timedelta
    
    doc_date = doc.metadata.get("date")
    if not doc_date:
        return True
    
    cutoff = datetime.now() - timedelta(days=30)
    return doc_date >= cutoff

retrieve_recent = create_filtered_retrieve_node(
    vectorstore,
    filter_func=only_recent_docs,
    k=4
)
```

---

### 3.2 生成节点（Generate Node）

#### 作用和职责

**核心职责**：基于检索到的文档生成答案

**输入**：
- `state["question"]`：用户问题
- `state["retrieved_docs"]`：检索到的文档

**输出**：
- `state["answer"]`：生成的答案
- `state["confidence_score"]`：答案置信度

---

#### 完整实现

```python
def create_generate_node(
    llm: BaseLLM,
    prompt_template: Optional[str] = None,
    include_sources: bool = False
):
    """
    创建生成节点
    
    Args:
        llm: LLM 实例
        prompt_template: 自定义 Prompt 模板
        include_sources: 是否在答案中包含来源
        
    Returns:
        生成节点函数
    """
    # 默认 Prompt 模板
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
        """生成节点：基于检索文档生成答案"""
        
        try:
            # ========== 步骤1: 获取问题 ==========
            question = state["question"]
            
            # ========== 步骤2: 获取并检查文档 ==========
            docs = state.get("retrieved_docs", [])
            
            if not docs:
                # 没有检索到文档，返回提示信息
                state["answer"] = "抱歉，我在知识库中没有找到相关信息来回答您的问题。"
                state["confidence_score"] = 0.0
                state["current_node"] = "generate"
                state["step_count"] = state.get("step_count", 0) + 1
                return state
            
            # ========== 步骤3: 组装上下文 ==========
            # 方式1: 简单拼接
            # context = "\n\n".join([doc.page_content for doc in docs])
            
            # 方式2: 添加编号（推荐）
            context = "\n\n".join([
                f"[文档 {i+1}]\n{doc.page_content}"
                for i, doc in enumerate(docs)
            ])
            
            # 方式3: 添加来源信息
            # context = "\n\n".join([
            #     f"[文档 {i+1} - 来源: {doc.metadata.get('source', '未知')}]\n{doc.page_content}"
            #     for i, doc in enumerate(docs)
            # ])
            
            # ========== 步骤4: 组装 Prompt ==========
            prompt = template.format(
                context=context,
                question=question
            )
            
            # ========== 步骤5: 调用 LLM 生成答案 ==========
            answer = llm.generate(prompt)
            
            # ========== 步骤6: 添加来源信息（可选）==========
            if include_sources:
                sources = list(set([
                    doc.metadata.get("source", "未知来源")
                    for doc in docs
                ]))
                answer += "\n\n参考来源：\n" + "\n".join(f"- {s}" for s in sources)
            
            # ========== 步骤7: 计算置信度 ==========
            retrieval_score = state.get("retrieval_score", 0.5)
            answer_length_score = min(len(answer) / 100, 1.0)
            
            confidence_score = (
                retrieval_score * 0.6 +      # 检索质量占60%
                answer_length_score * 0.4    # 答案长度占40%
            )
            
            # ========== 步骤8: 更新 State ==========
            state["answer"] = answer
            state["confidence_score"] = confidence_score
            state["current_node"] = "generate"
            state["step_count"] = state.get("step_count", 0) + 1
            
            return state
            
        except Exception as e:
            # 错误处理
            state["error"] = f"生成失败: {str(e)}"
            state["answer"] = f"抱歉，生成答案时出现错误。"
            state["confidence_score"] = 0.0
            return state
    
    return generate_node
```

---

#### 关键点解析

**1. 文档检查**

```python
if not docs:
    # 优雅降级：返回友好提示而不是抛出异常
    state["answer"] = "抱歉，我没有找到相关信息。"
    state["confidence_score"] = 0.0
    return state
```

**2. 上下文组装的多种方式**

```python
# 方式1: 简单拼接
context = "\n\n".join([doc.page_content for doc in docs])
# 输出:
# 文档1内容...
# 
# 文档2内容...

# 方式2: 添加编号（推荐）
context = "\n\n".join([
    f"[文档 {i+1}]\n{doc.page_content}"
    for i, doc in enumerate(docs)
])
# 输出:
# [文档 1]
# 文档1内容...
# 
# [文档 2]
# 文档2内容...

# 方式3: 添加来源
context = "\n\n".join([
    f"[文档 {i+1} - {doc.metadata.get('source', '未知')}]\n{doc.page_content}"
    for i, doc in enumerate(docs)
])
# 输出:
# [文档 1 - intro.txt]
# 文档1内容...
```

**3. Prompt 设计**

```python
# 基础版
template = "上下文：{context}\n问题：{question}\n答案："

# 专业版（推荐）
template = """你是一位专业助手。

上下文信息：
{context}

用户问题：
{question}

要求：
1. 仅基于上下文回答
2. 没有信息时明确说明
3. 回答要结构化

答案："""

# 高级版（带角色设定）
template = """你是{role}，擅长{expertise}。

上下文：
{context}

问题：{question}

要求：
{requirements}

答案："""
```

**4. 置信度计算**

```python
# 简单策略
confidence = retrieval_score * 0.6 + answer_length_score * 0.4

# 复杂策略
def calculate_confidence(state, answer):
    # 因素1: 检索质量
    retrieval_score = state.get("retrieval_score", 0.5)
    
    # 因素2: 答案长度
    length_score = min(len(answer) / 100, 1.0)
    
    # 因素3: 关键词匹配
    question_words = set(state["question"].lower().split())
    answer_words = set(answer.lower().split())
    keyword_score = len(question_words & answer_words) / len(question_words)
    
    # 因素4: 是否包含"不知道"类的表述
    uncertainty_phrases = ["不知道", "无法回答", "没有信息"]
    has_uncertainty = any(p in answer for p in uncertainty_phrases)
    uncertainty_penalty = 0.3 if has_uncertainty else 0
    
    # 综合计算
    confidence = (
        retrieval_score * 0.4 +
        length_score * 0.3 +
        keyword_score * 0.3 -
        uncertainty_penalty
    )
    
    return max(0, min(1, confidence))
```

---

#### 变体和扩展

**变体1: 流式生成节点**

```python
def create_stream_generate_node(llm, prompt_template=None):
    """流式生成节点"""
    template = prompt_template or get_default_template()
    
    def stream_generate_node(state):
        question = state["question"]
        docs = state.get("retrieved_docs", [])
        
        if not docs:
            state["answer"] = "没有找到相关信息。"
            return state
        
        context = "\n\n".join([doc.page_content for doc in docs])
        prompt = template.format(context=context, question=question)
        
        # 流式生成并收集
        answer_parts = []
        for chunk in llm.stream_generate(prompt):
            answer_parts.append(chunk)
            # 这里可以添加回调，实时输出
        
        state["answer"] = "".join(answer_parts)
        return state
    
    return stream_generate_node
```

**变体2: 多步推理生成节点**

```python
def create_chain_of_thought_generate_node(llm):
    """思维链生成节点"""
    def cot_generate_node(state):
        question = state["question"]
        docs = state.get("retrieved_docs", [])
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # 步骤1: 生成推理过程
        reasoning_prompt = f"""
        基于以下上下文，一步步分析如何回答问题：
        
        上下文：{context}
        问题：{question}
        
        请给出推理步骤：
        """
        reasoning = llm.generate(reasoning_prompt)
        state["reasoning_steps"] = reasoning
        
        # 步骤2: 基于推理生成答案
        answer_prompt = f"""
        推理过程：
        {reasoning}
        
        基于以上推理，请给出最终答案：
        """
        answer = llm.generate(answer_prompt)
        state["answer"] = answer
        
        return state
    
    return cot_generate_node
```

**变体3: 自我修正生成节点**

```python
def create_self_refine_generate_node(llm):
    """自我修正生成节点"""
    def self_refine_node(state):
        question = state["question"]
        docs = state.get("retrieved_docs", [])
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # 第一次生成
        initial_answer = llm.generate(f"""
        基于上下文回答：
        上下文：{context}
        问题：{question}
        答案：
        """)
        
        # 自我评估和修正
        refined_answer = llm.generate(f"""
        原始答案：{initial_answer}
        
        请评估以上答案并进行改进：
        1. 是否准确？
        2. 是否完整？
        3. 是否清晰？
        
        改进后的答案：
        """)
        
        state["initial_answer"] = initial_answer
        state["answer"] = refined_answer
        
        return state
    
    return self_refine_node
```

---

### 3.3 决策节点（Decide Node）

#### 作用和职责

**核心职责**：根据当前状态决定下一步操作

**特殊性**：
- 其他节点返回 `AgentState`
- 决策节点返回 `str`（下一个节点的名称）

**输入**：
- 整个 `AgentState`

**输出**：
- 字符串：下一个节点的名称（"retrieve", "generate", "end" 等）

---

#### 完整实现

```python
def decide_node(state: AgentState) -> str:
    """
    决策节点：决定 Agent 的下一步操作
    
    这是一个条件边（Conditional Edge）使用的函数。
    返回值是字符串，表示下一个要执行的节点名称。
    
    决策树：
    1. 检查终止条件
       - 达到最大步数 → "end"
       - 已有答案 → "end"
       - 有错误 → "end"
    
    2. 检查执行状态
       - 还未检索 → "retrieve"
       - 检索质量差 → "rewrite_query" 或 "retrieve"
       - 已检索但未生成 → "generate"
    
    Args:
        state: Agent 状态
        
    Returns:
        下一个节点的名称（字符串）
        
    可能的返回值：
        - "retrieve": 执行检索
        - "rewrite_query": 改写查询
        - "generate": 生成答案
        - "evaluate": 评估质量
        - "end": 结束执行
    """
    
    # ========== 决策1: 检查是否达到最大步数 ==========
    if state.get("step_count", 0) >= state.get("max_steps", 5):
        return "end"
    
    # ========== 决策2: 检查是否已有答案 ==========
    if state.get("answer"):
        return "end"
    
    # ========== 决策3: 检查是否有错误 ==========
    if state.get("error"):
        return "end"
    
    # ========== 决策4: 检查是否还未检索 ==========
    if state.get("retrieved_docs") is None:
        return "retrieve"
    
    # ========== 决策5: 检查检索质量 ==========
    retrieval_score = state.get("retrieval_score", 0)
    need_more_context = state.get("need_more_context", False)
    
    if need_more_context and retrieval_score < 0.5:
        # 检索质量很差，重写查询
        return "rewrite_query"
    
    # ========== 决策6: 检查是否已检索但未生成 ==========
    if state.get("retrieved_docs") and not state.get("answer"):
        return "generate"
    
    # ========== 默认：结束 ==========
    return "end"
```

---

#### 关键点解析

**1. 决策优先级**

```python
# 优先级从高到低：
1. 终止条件（最高优先级）
   - 达到最大步数
   - 已有答案
   - 有错误
   
2. 异常状态
   - 还未检索
   - 检索质量极差
   
3. 正常流程
   - 检索 → 生成
   
4. 默认处理
   - 结束

# 为什么这样排序？
# - 终止条件必须最先检查，防止无限循环
# - 异常要及时处理
# - 正常流程按步骤执行
```

**2. 返回值说明**

```python
# 返回值对应 Graph 中的节点名称

graph.add_conditional_edges(
    "decide",           # 从 decide 节点出发
    decide_node,        # 使用决策函数
    {
        "retrieve": "retrieve_node",      # "retrieve" → 检索节点
        "rewrite_query": "rewrite_node",  # "rewrite_query" → 改写节点
        "generate": "generate_node",      # "generate" → 生成节点
        "end": END                        # "end" → 结束
    }
)
```

**3. 状态检查的健壮性**

```python
# ✅ 使用 .get() 方法，提供默认值
if state.get("step_count", 0) >= state.get("max_steps", 5):
    return "end"

# ❌ 直接访问，可能KeyError
if state["step_count"] >= state["max_steps"]:
    return "end"

# 健壮性比较：
state1 = {}  # 空状态
# .get() 方式：返回默认值 0 和 5，正常比较
# 直接访问：KeyError!
```

---

#### 变体和扩展

**变体1: 复杂决策节点**

```python
def advanced_decide_node(state: AgentState) -> str:
    """高级决策节点：支持更多决策路径"""
    
    # 终止检查
    if state.get("step_count", 0) >= state.get("max_steps", 10):
        return "end"
    
    if state.get("error"):
        return "error_handler"  # 有专门的错误处理节点
    
    # 基于用户意图的决策
    question_type = classify_question(state["question"])
    
    if question_type == "factual":
        # 事实性问题：检索 → 生成
        if not state.get("retrieved_docs"):
            return "retrieve"
        else:
            return "generate"
    
    elif question_type == "analytical":
        # 分析性问题：检索 → 分析 → 生成
        if not state.get("retrieved_docs"):
            return "retrieve"
        elif not state.get("analysis"):
            return "analyze"
        else:
            return "generate"
    
    elif question_type == "creative":
        # 创造性问题：直接生成（无需检索）
        return "creative_generate"
    
    return "end"

def classify_question(question):
    """分类问题类型（简化示例）"""
    if any(word in question for word in ["是什么", "定义", "概念"]):
        return "factual"
    elif any(word in question for word in ["为什么", "如何", "原因"]):
        return "analytical"
    elif any(word in question for word in ["创作", "想象", "设计"]):
        return "creative"
    return "factual"
```

**变体2: 基于质量的决策**

```python
def quality_based_decide_node(state: AgentState) -> str:
    """基于质量分数的决策"""
    
    # 终止检查
    if state.get("step_count", 0) >= state.get("max_steps", 5):
        return "end"
    
    # 如果已有高质量答案，直接结束
    if state.get("answer") and state.get("confidence_score", 0) > 0.8:
        return "end"
    
    # 如果有答案但质量不高，尝试改进
    if state.get("answer") and state.get("confidence_score", 0) < 0.6:
        return "refine_answer"  # 答案改进节点
    
    # 检索质量评估
    retrieval_score = state.get("retrieval_score", 0)
    
    if not state.get("retrieved_docs"):
        return "retrieve"
    
    elif retrieval_score < 0.5:
        # 质量很差，重新检索
        if state.get("retrieval_attempts", 0) < 2:
            return "rewrite_and_retrieve"
        else:
            # 尝试过多次，降级处理
            return "fallback_answer"
    
    elif retrieval_score < 0.7:
        # 质量一般，增加检索数量
        return "retrieve_more"
    
    else:
        # 质量良好，生成答案
        return "generate"
```

**变体3: 带循环控制的决策**

```python
def loop_control_decide_node(state: AgentState) -> str:
    """带循环控制的决策节点"""
    
    # 最大步数检查
    if state.get("step_count", 0) >= state.get("max_steps", 10):
        return "force_end"
    
    # 检测循环
    execution_path = state.get("execution_path", [])
    
    # 如果连续3次执行同一个节点，跳出循环
    if len(execution_path) >= 3:
        last_three = execution_path[-3:]
        if len(set(last_three)) == 1:  # 3次都是同一个节点
            print(f"⚠️ 检测到循环：{last_three}")
            return "break_loop"
    
    # 正常决策逻辑
    if not state.get("retrieved_docs"):
        execution_path.append("retrieve")
        state["execution_path"] = execution_path
        return "retrieve"
    
    if not state.get("answer"):
        execution_path.append("generate")
        state["execution_path"] = execution_path
        return "generate"
    
    return "end"
```

---

## 四、最佳实践

### 4.1 错误处理

#### 原则

**不要让异常中断 Agent 执行**

```python
# ❌ 不好的做法 - 抛出异常
def retrieve_node(state):
    docs = vectorstore.search(state["question"])  # 可能抛出异常
    state["retrieved_docs"] = docs
    return state

# ✅ 好的做法 - 捕获异常并记录
def retrieve_node(state):
    try:
        docs = vectorstore.search(state["question"])
        state["retrieved_docs"] = docs
    except Exception as e:
        # 记录错误
        state["error"] = f"检索失败: {str(e)}"
        # 提供默认值
        state["retrieved_docs"] = []
    
    return state  # 确保总是返回 State
```

---

#### 分级错误处理

```python
def retrieve_node(state):
    try:
        docs = vectorstore.search(state["question"])
        state["retrieved_docs"] = docs
        
    except ConnectionError as e:
        # 连接错误：可能是临时的，标记需要重试
        state["error"] = f"连接失败: {str(e)}"
        state["error_type"] = "connection"
        state["should_retry"] = True
        state["retrieved_docs"] = []
        
    except ValueError as e:
        # 值错误：输入有问题，不应重试
        state["error"] = f"输入错误: {str(e)}"
        state["error_type"] = "validation"
        state["should_retry"] = False
        state["retrieved_docs"] = []
        
    except Exception as e:
        # 其他未知错误
        state["error"] = f"未知错误: {str(e)}"
        state["error_type"] = "unknown"
        state["should_retry"] = False
        state["retrieved_docs"] = []
    
    return state
```

---

#### 带重试的错误处理

```python
def create_retrieve_with_retry(vectorstore, max_retries=3):
    """带重试的检索节点"""
    def retrieve_node(state):
        for attempt in range(max_retries):
            try:
                docs = vectorstore.search(state["question"])
                state["retrieved_docs"] = docs
                state["error"] = None  # 清除之前的错误
                return state
                
            except Exception as e:
                if attempt == max_retries - 1:
                    # 最后一次尝试也失败了
                    state["error"] = f"重试{max_retries}次后仍失败: {str(e)}"
                    state["retrieved_docs"] = []
                    return state
                
                # 等待后重试
                import time
                time.sleep(2 ** attempt)  # 指数退避
        
        return state
    
    return retrieve_node
```

---

### 4.2 日志记录

#### 结构化日志

```python
def retrieve_node(state):
    """带结构化日志的检索节点"""
    import logging
    from datetime import datetime
    
    # 初始化日志列表
    if "processing_log" not in state:
        state["processing_log"] = []
    
    # 记录开始
    start_time = datetime.now()
    state["processing_log"].append({
        "node": "retrieve",
        "action": "start",
        "timestamp": start_time.isoformat(),
        "question": state["question"]
    })
    
    try:
        # 执行检索
        docs = vectorstore.search(state["question"])
        state["retrieved_docs"] = docs
        
        # 记录成功
        state["processing_log"].append({
            "node": "retrieve",
            "action": "success",
            "timestamp": datetime.now().isoformat(),
            "doc_count": len(docs),
            "duration": (datetime.now() - start_time).total_seconds()
        })
        
    except Exception as e:
        # 记录失败
        state["processing_log"].append({
            "node": "retrieve",
            "action": "error",
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
            "duration": (datetime.now() - start_time).total_seconds()
        })
        state["error"] = str(e)
        state["retrieved_docs"] = []
    
    return state
```

---

#### 性能监控

```python
def retrieve_node(state):
    """带性能监控的节点"""
    import time
    
    # 初始化计时字典
    if "node_timings" not in state:
        state["node_timings"] = {}
    
    start_time = time.time()
    
    try:
        # 执行业务逻辑
        docs = vectorstore.search(state["question"])
        state["retrieved_docs"] = docs
        
    finally:
        # 记录耗时（无论成功失败）
        elapsed = time.time() - start_time
        state["node_timings"]["retrieve"] = elapsed
        
        # 性能警告
        if elapsed > 2.0:  # 超过2秒
            print(f"⚠️ [性能警告] retrieve 节点耗时 {elapsed:.2f}s")
    
    return state
```

---

### 4.3 步骤计数

#### 基础步骤计数

```python
def any_node(state):
    """任何节点都应该更新步骤计数"""
    
    # 执行业务逻辑
    # ...
    
    # 更新步骤计数
    state["step_count"] = state.get("step_count", 0) + 1
    
    # 记录当前节点
    state["current_node"] = "node_name"
    
    return state
```

---

#### 详细执行路径

```python
def any_node(state):
    """记录详细的执行路径"""
    
    # 初始化执行路径
    if "execution_path" not in state:
        state["execution_path"] = []
    
    # 记录节点执行
    from datetime import datetime
    state["execution_path"].append({
        "node": "retrieve",
        "step": state.get("step_count", 0) + 1,
        "timestamp": datetime.now().isoformat()
    })
    
    # 更新步骤计数
    state["step_count"] = state.get("step_count", 0) + 1
    
    # 执行业务逻辑
    # ...
    
    return state
```

---

### 4.4 状态验证

#### 输入验证

```python
def generate_node(state):
    """带输入验证的生成节点"""
    
    # 验证必需字段
    required_fields = ["question", "retrieved_docs"]
    missing_fields = [f for f in required_fields if f not in state]
    
    if missing_fields:
        state["error"] = f"缺少必需字段: {', '.join(missing_fields)}"
        return state
    
    # 验证字段类型
    if not isinstance(state["retrieved_docs"], list):
        state["error"] = "retrieved_docs 必须是列表"
        return state
    
    # 验证字段值
    if len(state["retrieved_docs"]) == 0:
        state["error"] = "retrieved_docs 不能为空"
        return state
    
    # 验证通过，执行业务逻辑
    # ...
    
    return state
```

---

#### 输出验证

```python
def generate_node(state):
    """带输出验证的生成节点"""
    
    # 执行业务逻辑
    answer = llm.generate(...)
    
    # 验证输出
    if not answer or len(answer.strip()) == 0:
        state["error"] = "生成的答案为空"
        state["answer"] = "抱歉，生成答案失败。"
        return state
    
    if len(answer) < 10:
        # 答案太短，可能有问题
        state["warning"] = "生成的答案可能不完整"
    
    # 保存答案
    state["answer"] = answer
    
    return state
```

---

## 五、常见问题

### Q1: 节点可以调用其他节点吗？

**A**: 技术上可以，但强烈不推荐。应该由 Graph 控制流程。

```python
# ❌ 不推荐 - 节点间直接调用
def combined_node(state):
    state = retrieve_node(state)  # 直接调用另一个节点
    state = generate_node(state)
    return state

# ✅ 推荐 - 由 Graph 控制
graph.add_edge("retrieve", "generate")
# Graph 会自动按顺序执行
```

**原因**：
- 破坏了节点的独立性
- 难以调试和测试
- Graph 无法追踪执行流程

---

### Q2: 如何在节点间共享数据？

**A**: 通过 State。所有共享数据都应该放在 State 中。

```python
# ✅ 通过 State 共享
def retrieve_node(state):
    docs = vectorstore.search(state["question"])
    state["retrieved_docs"] = docs  # 保存到 State
    return state

def generate_node(state):
    docs = state["retrieved_docs"]  # 从 State 读取
    answer = llm.generate(docs)
    return state

# ❌ 使用全局变量（不推荐）
global_docs = None

def retrieve_node(state):
    global global_docs
    global_docs = vectorstore.search(state["question"])
    return state
```

---

### Q3: 节点必须返回 State 吗？

**A**: 普通节点必须返回 State，决策节点返回字符串。

```python
# 普通节点 - 返回 AgentState
def retrieve_node(state: AgentState) -> AgentState:
    # ...
    return state

# 决策节点 - 返回 str
def decide_node(state: AgentState) -> str:
    # ...
    return "next_node"
```

---

### Q4: 可以只返回部分更新的 State 吗？

**A**: 可以。LangGraph 会自动合并更新。

```python
# 方式1: 返回完整 State
def node1(state):
    state["answer"] = "..."
    return state  # 返回整个 state

# 方式2: 只返回更新的字段
def node2(state):
    return {"answer": "..."}  # LangGraph 自动合并

# 两种方式等效
```

---

### Q5: 如何处理节点执行失败？

**A**: 捕获异常，记录到 State，不要中断流程。

```python
def safe_node(state):
    try:
        # 执行业务逻辑
        result = risky_operation()
        state["result"] = result
        
    except Exception as e:
        # 记录错误，不抛出异常
        state["error"] = str(e)
        state["result"] = None  # 提供默认值
    
    return state  # 确保总是返回
```

---

### Q6: 节点可以有副作用吗？

**A**: 尽量避免，但某些情况可以接受（如日志、监控）。

```python
# ✅ 可接受的副作用
def node_with_logging(state):
    # 写入日志文件
    logger.info(f"Processing: {state['question']}")
    
    # 发送监控指标
    metrics.increment("node.retrieve.calls")
    
    # 执行业务逻辑
    # ...
    
    return state

# ❌ 不好的副作用
def bad_node(state):
    # 修改全局状态
    global global_cache
    global_cache[state["question"]] = result
    
    # 修改数据库
    db.update(...)
    
    return state
```

---

### Q7: 如何测试节点？

**A**: 使用 mock 对象，独立测试每个节点。

```python
def test_retrieve_node():
    # 1. 创建 mock 对象
    mock_vectorstore = MockVectorStore()
    mock_vectorstore.set_return([doc1, doc2])
    
    # 2. 创建节点
    retrieve_node = create_retrieve_node(mock_vectorstore, k=2)
    
    # 3. 准备输入
    state = {"question": "test"}
    
    # 4. 执行节点
    result = retrieve_node(state)
    
    # 5. 验证输出
    assert "retrieved_docs" in result
    assert len(result["retrieved_docs"]) == 2
    assert result["step_count"] == 1
```

---

### Q8: 节点可以是异步的吗？

**A**: 可以。LangGraph 支持异步节点。

```python
async def async_retrieve_node(state):
    """异步检索节点"""
    docs = await async_vectorstore.search(state["question"])
    state["retrieved_docs"] = docs
    return state

# 在异步 Graph 中使用
graph = StateGraph(AgentState)
graph.add_node("retrieve", async_retrieve_node)
```

---

## 六、总结

### 核心要点

1. **节点是纯函数**
   - 输入：AgentState
   - 输出：AgentState（或字符串用于决策节点）
   - 无副作用

2. **职责单一**
   - 每个节点只做一件事
   - 便于测试、维护、复用

3. **使用工厂模式**
   - 注入外部依赖
   - 参数化配置
   - 便于测试

4. **完善的错误处理**
   - 不要抛出异常
   - 记录错误到 State
   - 提供默认值

5. **详细的日志**
   - 记录执行路径
   - 性能监控
   - 便于调试

### 节点分类

| 类型 | 作用 | 返回值 | 示例 |
|------|------|--------|------|
| 数据处理 | 处理转换数据 | AgentState | retrieve, parse |
| 生成节点 | 调用 LLM | AgentState | generate, summarize |
| 决策节点 | 控制流程 | str | decide |
| 评估节点 | 评估质量 | AgentState | evaluate |
| 工具调用 | 调用外部工具 | AgentState | web_search |
| 辅助节点 | 辅助功能 | AgentState | log, format |

### 设计模式

| 模式 | 优势 | 适用场景 |
|------|------|----------|
| 工厂模式 | 依赖注入、便于测试 | 所有需要外部依赖的节点 |
| 类方法 | 共享配置、组织清晰 | 大型项目、复杂 Agent |
| 装饰器 | 关注点分离、代码复用 | 添加横切关注点 |

### 最佳实践

- ✅ 使用类型注解
- ✅ 完善的文档字符串
- ✅ 错误处理不抛异常
- ✅ 记录详细日志
- ✅ 更新步骤计数
- ✅ 验证输入输出
- ✅ 编写单元测试

### 下一步

学习**图构建（Graph）**，将节点连接起来形成完整的 Agent！

---

**文档版本**: 1.0  
**创建日期**: 2026-01-13  
**适用项目**: HuahuaChat 阶段三
