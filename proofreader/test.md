# 面临的挑战

虽然生成式AI技术及工具已经在软件生命周期的各个环节中体现出了提效的可行性，但想要在大型科技组织中端到端落地、实现提效的突破，还面临很多挑战。

## 企业规模化软件过程提效的挑战

**信息安全和信创化的强制要求**

大型企业尤其是银行，面临最强的信息安全监管及信创化、国产化要求，需要国产生态中能力突出的大模型。

**开源大模型表现偏弱、自己训练成本高**

目前可私有化部署的大模型，其自然语言理解和代码生成的水平与GPT有一定差距；根据大语言模型论文做一些粗略的数学计算，如果用的是
Facebook LLaMA，训练成本（不考虑迭代或者出错）大约是400 万美元，如果是谷歌PaLM，大约 2700 万美元。

**与企业内部工具进行结合**

碎片化的应用提效效果有限，把 AI 无缝集成到BizDevOps 工具链中，技术难度尚未可知。

## 开发 AI 辅助研发提效的局限性

**基于 GPT 模型的工具不符合信息安全要求**

目前大多工具基于 OpenAI GPT 构建，不能私有化部署，不符合信息安全的强制要求；需要寻找能够私有化部署且水平相当的替代品。

**公开LLM 针对专业领域表现不佳，适用性差**

缺乏专业知识，对于某些专业领域缺乏足够的理解。它受训练数据中的影响，容易产生偏见和错误。

**LLM 工具碎片化**

各类工具都是在一个个分散的工作节点上辅助，使用时来回切换工具的成本很高，整体端到端地提效不明显。
 

# LLM 应用示例：最佳实践示例

## LLM 应用开发模式：轻量级 API 编排

在 LangChain 中使用了思维链的方式来选择合适的智能体（Agent），在 Co-mate 中，我们也是采取了类似的设计，在本地构建好函数，然后交由
LLM 来分析用户的输入适合调用哪个函数。

如下是我们的 prompt 示例：

```
Answer the following questions as best you can. You have access to the following tools:

introduce_system: introduce_system is a function to introduce a system.

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [introduce_system]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: Introduce the following system: https://github.com/archguard/ddd-monolithic-code-sample
```

这里的 `Question` 便是用户的输入，然后再调用对应的 `introduce_system` 函数进行分析。

## LLM 应用开发模式：DSL 动态运行时 

与事实能力相比，我们更信任 LLM 的编排能力，因此我们在 Co-mate 中采用了 DSL 的方式来编排函数，这样可以更加灵活的编排函数。

为了支撑这样的能力，我们在 Co-mate 中引入了 Kotlin 作为 DSL 的运行时：

```kotlin
// 初始化运行时
val repl = KotlinInterpreter()
val mvcDslSpec = repl.evalCast<FoundationSpec>(InterpreterRequest(code = mvcFoundation))

// 从用户的输入中获取 action
val action = ComateToolingAction.from(action.lowercase())

// 添加默认的 DSL spec
if (action == ComateToolingAction.FOUNDATION_SPEC_GOVERNANCE) {
    comateContext.spec = mvcDslSpec
}
```

对应的 DSL 示例（由 ChatGPT 根据 DDD 版本 spec 生成）：

```kotlin
foundation {
    project_name {
        pattern("^([a-z0-9-]+)-([a-z0-9-]+)(-common)?\${'$'}")
        example("system1-webapp1")
    }

    layered {
        layer("controller") {
            pattern(".*\\.controller") { name shouldBe endsWith("Controller") }
        }
        layer("service") {
            pattern(".*\\.service") {
                name shouldBe endsWith("DTO", "Request", "Response", "Factory", "Service")
            }
        }
        layer("repository") {
            pattern(".*\\.repository") { name shouldBe endsWith("Entity", "Repository", "Mapper") }
        }

        dependency {
            "controller" dependedOn "service"
            "controller" dependedOn "repository"
            "service" dependedOn "repository"
        }
    }

    naming {
        class_level {
            style("CamelCase")
            pattern(".*") { name shouldNotBe contains("${'$'}") }
        }
        function_level {
            style("CamelCase")
            pattern(".*") { name shouldNotBe contains("${'$'}") }
        }
    }
}
```

## LLM 应用开发模式：本地小模型

在 Co-mate 中，我们在本地引入了 SentenceTransformer 来处理用户的输入，优在本地分析、匹配用户的输入，并处理。当匹配到结果后直接调用本地的函数，当匹配不到结果时调用远端的处理函数来处理。

HuggingFace: [https://huggingface.co/sentence-transformers](https://huggingface.co/sentence-transformers)

在原理上主要是参考了 GitHub Copilot、 Bloop 的实现，通过本地的小模型来处理用户的输入，然后再通过远端的大模型来处理用户的输入。

### Rust 实现示例

Rust 相关示例：[https://github.com/unit-mesh/unit-agent](https://github.com/unit-mesh/unit-agent)

```rust
pub fn embed(&self, sequence: &str) -> anyhow::Result<Embedding> {
    let tokenizer_output = self.tokenizer.encode(sequence, true).unwrap();

    let input_ids = tokenizer_output.get_ids();
    let attention_mask = tokenizer_output.get_attention_mask();
    let token_type_ids = tokenizer_output.get_type_ids();
    let length = input_ids.len();
    trace!("embedding {} tokens {:?}", length, sequence);

    let inputs_ids_array = ndarray::Array::from_shape_vec(
        (1, length),
        input_ids.iter().map(|&x| x as i64).collect(),
    )?;

    let attention_mask_array = ndarray::Array::from_shape_vec(
        (1, length),
        attention_mask.iter().map(|&x| x as i64).collect(),
    )?;

    let token_type_ids_array = ndarray::Array::from_shape_vec(
        (1, length),
        token_type_ids.iter().map(|&x| x as i64).collect(),
    )?;

    let outputs = self.session.run([
        InputTensor::from_array(inputs_ids_array.into_dyn()),
        InputTensor::from_array(attention_mask_array.into_dyn()),
        InputTensor::from_array(token_type_ids_array.into_dyn()),
    ])?;

    let output_tensor: OrtOwnedTensor<f32, _> = outputs[0].try_extract().unwrap();
    let sequence_embedding = &*output_tensor.view();
    let pooled = sequence_embedding.mean_axis(Axis(1)).unwrap();

    Ok(pooled.to_owned().as_slice().unwrap().to_vec())
}
```

### Kotlin 实现示例

```kotlin
class Semantic(val tokenizer: HuggingFaceTokenizer, val session: OrtSession, val env: OrtEnvironment) {
    fun embed(
        sequence: String,
    ): FloatArray {
        val tokenized = tokenizer.encode(sequence, true)

        val inputIds = tokenized.ids
        val attentionMask = tokenized.attentionMask
        val typeIds = tokenized.typeIds

        val tensorInput = OrtUtil.reshape(inputIds, longArrayOf(1, inputIds.size.toLong()))
        val tensorAttentionMask = OrtUtil.reshape(attentionMask, longArrayOf(1, attentionMask.size.toLong()))
        val tensorTypeIds = OrtUtil.reshape(typeIds, longArrayOf(1, typeIds.size.toLong()))

        val result = session.run(
            mapOf(
                "input_ids" to OnnxTensor.createTensor(env, tensorInput),
                "attention_mask" to OnnxTensor.createTensor(env, tensorAttentionMask),
                "token_type_ids" to OnnxTensor.createTensor(env, tensorTypeIds),
            ),
        )

        val outputTensor = result.get(0) as OnnxTensor
        val output = outputTensor.floatBuffer.array()

        return output
    }


    companion object {
        fun create(): Semantic {
            val classLoader = Thread.currentThread().getContextClassLoader()

            val tokenizerPath = classLoader.getResource("model/tokenizer.json")!!.toURI()
            val onnxPath =  classLoader.getResource("model/model.onnx")!!.toURI()

            try {
                val env: Map<String, String> = HashMap()
                val array: List<String> = tokenizerPath.toString().split("!")
                FileSystems.newFileSystem(URI.create(array[0]), env)
            } catch (e: Exception) {
//                e.printStackTrace()
            }

            val tokenizer = HuggingFaceTokenizer.newInstance(Paths.get(tokenizerPath))
            val ortEnv = OrtEnvironment.getEnvironment()
            val sessionOptions = OrtSession.SessionOptions()

            // load onnxPath as byte[]
            val onnxPathAsByteArray = Files.readAllBytes(Paths.get(onnxPath))

            val session = ortEnv.createSession(onnxPathAsByteArray, sessionOptions)

            return Semantic(tokenizer, session, ortEnv)
        }
    }
}
```

## LLM 应用开发模式：Stream 封装

### 服务端 API 调用：Kotlin 实现

机制：结合 callbackFlow 来实现

```kotlin
fun stream(text: String): Flow<String> {
    val systemMessage = ChatMessage(ChatMessageRole.USER.value(), text)

    messages.add(systemMessage)

    val completionRequest = ChatCompletionRequest.builder()
        .model(openAiVersion)
        .temperature(0.0)
        .messages(messages)
        .build()

    return callbackFlow {
        withContext(Dispatchers.IO) {
            service.streamChatCompletion(completionRequest)
                .doOnError(Throwable::printStackTrace)
                .blockingForEach { response ->
                    val completion = response.choices[0].message
                    if (completion != null && completion.content != null) {
                        trySend(completion.content)
                    }
                }

            close()
        }
    }
}
```

### 客户端 API 调用：TypeScript 实现

机制：依赖于 Vercel 的 AI 库，提供对于 Stream 的封装

```typescript
import { Message, OpenAIStream, StreamingTextResponse } from 'ai'
import { Configuration, OpenAIApi } from 'openai-edge'

export async function stream(apiKey: string, messages: Message[], isStream: boolean = true) {
  let basePath = process.env.OPENAI_PROXY_URL
  if (basePath == null) {
    basePath = 'https://api.openai.com'
  }

  const configuration = new Configuration({
    apiKey: apiKey || process.env.OPENAI_API_KEY,
    basePath
  })

  const openai = new OpenAIApi(configuration)

  const res = await openai.createChatCompletion({
    model: 'gpt-3.5-turbo',
    messages,
    temperature: 0.7,
    stream: isStream
  })

  if (!isStream) {
    return res
  }

  const stream = OpenAIStream(res, {})

  return new StreamingTextResponse(stream)
}
```

### 客户端 UI 实现：Fetch

```typescript
const decoder = new TextDecoder()

export function decodeAIStreamChunk(chunk: Uint8Array): string {
  return decoder.decode(chunk)
}

await fetch("/api/action/tooling", {
  method: "POST",
  body: JSON.stringify(tooling),
}).then(async response => {
  onResult(await response.json())
  let result = ""
  const reader = response.body.getReader()
  while (true) {
    const { done, value } = await reader.read()
    if (done) {
      break
    }

    result += decodeAIStreamChunk(value)
    onResult(result)
  }

  isPending = false
});
```

### 服务端实现转发： Java + Spring

WebFlux + Spring Boot

```java
@RestController
public class ChatController {

    private WebClient webClient = WebClient.create();

    @PostMapping(value = "/api/chat", produces = MediaType.TEXT_EVENT_STREAM_VALUE)
    public SseEmitter chat(@RequestBody ChatInput input) throws IOException {
        SseEmitter emitter = new SseEmitter();

        webClient.post()
                .uri(REMOTE_URL)
                .bodyValue(input)
                .exchangeToFlux(response -> {
                    if (response.statusCode().is2xxSuccessful()) {
                        return response.bodyToFlux(byte[].class)
                                .map(String::new)
                                .doOnNext(string -> {
                                    try {
                                        emitter.send(string);
                                    } catch (IOException e) {
                                        logger.error("Error while sending data: {}", e.getMessage());
                                        emitter.completeWithError(e);
                                    }
                                })
                                .doOnComplete(emitter::complete)
                                .doOnError(emitter::completeWithError);
                    } else {
                        emitter.completeWithError(new RuntimeException("Error while calling remote service"));
                    }
                })
                .subscribe();

        return emitter;
    }
}
```

### 服务端转发：Python

FastAPI + OpenAI

```python
def generate_reply_stream(input_data: ChatInput):
    prompt = input_data.message
    try:
        prompt = prompt
        response = openai.ChatCompletion.create(
            model=openai_model,
            temperature=temperature,
            max_tokens=max_tokens,
            n=max_responses,
            messages=[
                {"role": "user", "content": prompt},
            ],
            stream=True,
        )
    except Exception as e:
        print("Error in creating campaigns from openAI:", str(e))
        raise HTTPException(503, error503)
    try:
        for chunk in response:
            current_content = chunk["choices"][0]["delta"].get("content", "")
            yield current_content

    except Exception as e:
        print("OpenAI Response (Streaming) Error: " + str(e))
        raise HTTPException(503, error503)


@app.post("/api/chat", response_class=Response)
async def chat(input_data: ChatInput):
    return StreamingResponse(generate_reply_stream(input_data), media_type="text/event-stream")
```

# LLM 应用开发模式：精细化流程

一个结合 AI 的自动化的工作流应该包含四个要素：

- 场景模板化，即预设各种常见的工作场景，为用户提供快捷的开始工作的方式。
- 交互式环境，包括但不限于输入框、按钮、编辑器、错误信息、帮助文档等，使用户能够与系统进行交互和反馈。
- 格式化输出，为用户提供规范的输出结果，避免信息过载或无用信息。
- 流程与工具集成，将不同的工具和流程集成到一个自动化的流程中，提高工作效率和准确性。同时，通过 AI 技术的支持，让系统能够智能化地处理数据和信息，进一步提高工作效率和准确性。

简单来说，就是我们依旧需要碳基生物作为检查官，来检查硅基生物输出是否合理？

## 设计构建高质量流程

基于我们对 ChatGPT 的使用经验，我们建议在使用 ChatGPT 之前，先考虑如何设计高质量的流程。这里的高质量流程，指的是：

- 明确的目标和目的：高质量的流程应该有明确的目标和目的，确保流程的设计和执行都能够达成预期的结果。 
- 易于理解和操作：高质量的流程应该简单易懂，让执行者能够轻松理解并操作。流程的设计应该尽可能避免复杂的步骤和冗长的说明，以免造成执行者的困惑和错误。
- 明确的责任和角色：高质量的流程应该明确各个执行者的责任和角色，避免执行者的混淆和错误。流程设计者应该明确各个角色的职责和权限，确保流程的顺利执行。
- 可度量和评估：高质量的流程应该能够被度量和评估。流程设计者应该设计合适的指标和评估方式，以便对流程的执行效果进行评估和改进。

如下是我们对于 SDLC + LLM 的探索过程中的展示示例：

![SDLC](images/llm-sdlc-processes.png)

将旅程拆得足够的详细，才能获得最好的效果。

## ChatFlow 的诞生动机：人类设计高质量流程 + AI 完成细节

在我使用了 ChatGPT （GPT 3.5）一个月多月之后，大抵算是掌握了它的脾气。简单来说，ChatGPT 即是一个硅基生物，也是一个非常好的人类助手。作为一个工具，你使用 prompt 的能力决定了它的上限和下限。

简单来说，ChatGPT 在经验丰富的开发人员手中，有一定 prompt 经历的人手中，会发挥非常强大的作用。而对于经验不那么丰富的开发人员，可能会因为缺乏任务分解能力，无法写出合理地 prompt，让 AI 有创意地瞎写代码。

诸如于，我们可以通过如下的注释，让 ChatGPT 或者 GitHub Copilot 直接生成可用的代码：

```jsx
// 1. convert resources in src/assets/chatgpt/category/*.yml to json
// 2. generate src/assets/chatgpt/category.json
// the yaml file is like this:
// ```yml
// ···
```

这也就是为什么我们做了 [ClickPrompt]([https://github.com/prompt-engineering/click-prompt](https://github.com/prompt-engineering/click-prompt)) ， 用于一键轻松查看、分享和执行 Prompt。而在完善 ClickPrompt 的过程中，我们发现将 AI 绑定到自己的工作流中，才能更好地提升效率。因此，我们在 ClickPrompt 中提取了两个功能，构建了 ChatFlow：

- ChatGPT 聊天室：一个集成了 ChatGPT API 的简易 ChatGPT聊天室。
- ClickFlow：一个基于 Yaml 构建的工作流。

结合 ClickPrompt 不丰富的组件，它可以勉强 work 了。

## ChatFlow 是什么？

![ChatFlow](images/chatflow-writing.png)

ChatFlow 是一个基于自然语言处理（NLP）的流程编排工具，具有以下特点：

- 易于理解的 YAML：ChatFlow 使用简单易懂的 YAML 格式来描述流程的各个元素，包括条件、循环和变量等。无需编程技能，让流程设计变得简单易懂。
- 丰富的可视化组件：ChatFlow 提供了丰富的可视化组件，例如表格、图表和交互式界面等，让用户可以更加方便地与流程进行交互和管理。
- 自动化执行流程：ChatFlow 使用 NLP 技术自动翻译自然语言描述的流程为可执行的代码，并支持自定义函数和自动生成文档功能，让用户更加灵活和高效地管理流程。

总之，ChatFlow 提供了一种灵活、易用、自动化的流程编排工具，让用户可以更加高效地管理复杂的流程，提高工作效率和准确性，同时降低工作的复杂性和学习成本。

## ChatFlow 示例

在过去的一段时间内，我们不断尝试开发一些工作流：

1. 需求与代码生成：从一个模糊的需求开始，生成标准的用户 Story（包含多个 AC），然后根据 AC 生成流程图、测试用例和测试代码。
2. 软件系统设计：从一个简单的系统开始，分析系统对应的用户旅程，生成对应的处理过程 DSL 等等，以帮助我们思考如何基于 AI 进行系统设计。
3. 写作的发散与探索：从一个主题开始，进行对应的发散和收敛，直至辅助我们完成一篇文章的草稿、大纲、内容编写。
4. ClickPrompt 工作流：围绕 ClickPrompt 项目的开发，结合创建 issue、issue 分析、Code Review 等构建的工作流。

在线示例：[https://www.clickprompt.org/zh-CN/click-flow/](https://www.clickprompt.org/zh-CN/click-flow/)

### ChatFlow 示例：需求与代码生成。

用于帮助开发人员快速生成代码并进行测试，从而加快开发进度和提高代码质量。

![](images/chatflow-ac.png)

### ChatFlow 示例：软件系统设计

用于帮助系统设计人员快速理解用户需求并生成对应的系统设计方案。

![](images/chatflow-software-design.png)

### ChatFlow 示例：写作的发散与探索

用于帮助写作人员快速生成文章并进行修改和编辑，从而提高写作效率和文章质量。

![ChatFlow](images/chatflow-writing.png)

### ChatFlow 示例：ClickPrompt 工作流

用于帮助开发团队快速解决问题并进行代码审查，从而加快项目进度和提高代码质量。

![](images/clickprompt-workflow.png)

