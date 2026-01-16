# ChatOpenAI 实现详解

## 一、关键知识点

### 1. ChatOpenAI 的调用方式

**重要**：`ChatOpenAI` 不是直接调用 `generate()` 方法，而是使用：
- `invoke()` - 同步调用，返回完整结果
- `stream()` - 流式调用，返回迭代器

### 2. 消息格式

`ChatOpenAI` 使用**消息列表**，而不是简单的字符串。需要导入 `HumanMessage`：

```python
from langchain_core.messages import HumanMessage

# 将字符串转换为消息
message = HumanMessage(content="你的问题")
messages = [message]

# 调用
response = llm.invoke(messages)
```

### 3. 返回格式

`invoke()` 返回的是 `AIMessage` 对象，需要提取 `content` 属性：

```python
response = llm.invoke(messages)
answer = response.content  # 提取文本内容
```

## 二、generate() 方法实现步骤

### 步骤1：导入必要的模块

```python
from langchain_core.messages import HumanMessage
```

### 步骤2：将字符串转换为消息

```python
# prompt 是字符串，需要转换为 HumanMessage
message = HumanMessage(content=prompt)
messages = [message]
```

### 步骤3：调用 ChatOpenAI

```python
# 使用 self.llm.invoke() 方法
response = self.llm.invoke(messages)
```

### 步骤4：提取内容

```python
# response 是 AIMessage 对象，需要提取 content
answer = response.content
return answer
```

### 完整实现思路（伪代码）

```python
def generate(self, prompt: str, **kwargs) -> str:
    # 1. 将字符串 prompt 转换为 HumanMessage
    # 2. 创建消息列表 [message]
    # 3. 调用 self.llm.invoke(messages)
    # 4. 从返回的 AIMessage 中提取 content
    # 5. 返回字符串
```

## 三、stream_generate() 方法实现步骤

### 关键点

`stream()` 方法返回一个迭代器，每次迭代返回一个 `AIMessageChunk` 对象。

### 步骤1：将字符串转换为消息

```python
message = HumanMessage(content=prompt)
messages = [message]
```

### 步骤2：使用 stream() 方法

```python
# stream() 返回迭代器
for chunk in self.llm.stream(messages):
    # chunk 是 AIMessageChunk 对象
    # 需要提取 content
    yield chunk.content
```

### 完整实现思路（伪代码）

```python
def stream_generate(self, prompt: str, **kwargs) -> Iterator[str]:
    # 1. 将字符串 prompt 转换为 HumanMessage
    # 2. 创建消息列表 [message]
    # 3. 使用 self.llm.stream(messages) 获取迭代器
    # 4. 遍历迭代器，提取每个 chunk 的 content
    # 5. 使用 yield 返回每个文本片段
```

## 四、常见问题

### Q1: 为什么不能用 llm.generate()？

**A**: `ChatOpenAI` 是 LangChain 的 Chat Model，使用 `invoke()` 或 `stream()` 方法。`generate()` 是旧版 LLM 接口的方法。

### Q2: 为什么要用 HumanMessage？

**A**: Chat Model 使用消息格式，支持多轮对话。消息类型包括：
- `HumanMessage` - 用户消息
- `AIMessage` - AI 回复
- `SystemMessage` - 系统消息

### Q3: 如何处理错误？

**A**: 可以添加 try-except：

```python
try:
    response = self.llm.invoke(messages)
    return response.content
except Exception as e:
    raise Exception(f"生成失败: {str(e)}")
```

## 五、参考文档链接

1. **ChatOpenAI 文档**：
   https://python.langchain.com/docs/integrations/chat/openai

2. **消息类型文档**：
   https://python.langchain.com/docs/modules/model_io/chat/

3. **调用方法文档**：
   在 ChatOpenAI 文档中查找 "invoke" 和 "stream"

## 六、测试建议

实现后，可以这样测试：

```python
from src.core.llm.base import OpenAILLM

# 创建实例
llm = OpenAILLM(model_name="gpt-3.5-turbo", temperature=0.7)

# 测试 generate
answer = llm.generate("你好，请介绍一下自己")
print(f"回答: {answer}")

# 测试 stream_generate
print("流式回答: ", end="")
for chunk in llm.stream_generate("请数1到5"):
    print(chunk, end="", flush=True)
print()  # 换行
```

## 七、完整代码结构参考

```python
class OpenAILLM(BaseLLM):
    def __init__(self, model_name: str, temperature: float = 0.7):
        super().__init__(model_name, temperature)
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=os.getenv("OPENAI_API_KEY"),
        )
    
    def generate(self, prompt: str, **kwargs) -> str:
        # TODO: 你的实现
        # 提示：
        # 1. 导入 HumanMessage
        # 2. 将 prompt 转换为 HumanMessage
        # 3. 调用 self.llm.invoke([message])
        # 4. 返回 response.content
        pass
    
    def stream_generate(self, prompt: str, **kwargs) -> Iterator[str]:
        # TODO: 你的实现
        # 提示：
        # 1. 导入 HumanMessage
        # 2. 将 prompt 转换为 HumanMessage
        # 3. 使用 self.llm.stream([message])
        # 4. 遍历并 yield chunk.content
        pass
```

现在你可以根据这些提示完成实现了！

