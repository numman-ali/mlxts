import { afterEach, describe, expect, test } from "bun:test";
import { mkdtempSync, rmSync, writeFileSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";

import { loadChatTemplate } from "./chat-template";

const tempRoots: string[] = [];

function createTempDir(prefix: string): string {
  const directory = mkdtempSync(join(tmpdir(), prefix));
  tempRoots.push(directory);
  return directory;
}

afterEach(() => {
  while (tempRoots.length > 0) {
    const directory = tempRoots.pop();
    if (directory !== undefined) {
      rmSync(directory, { recursive: true, force: true });
    }
  }
});

describe("loadChatTemplate", () => {
  test("loads a chat template from tokenizer_config.json", async () => {
    const directory = createTempDir("mlxts-transformers-chat-template-");
    writeFileSync(join(directory, "config.json"), JSON.stringify({ model_type: "llama" }));
    writeFileSync(
      join(directory, "tokenizer_config.json"),
      JSON.stringify({
        bos_token: "<s>",
        eos_token: "</s>",
        chat_template:
          "{{ bos_token }}{% for message in messages %}[{{ message['role'] }}] {{ message['content'] }}{% endfor %}{% if add_generation_prompt %}[assistant] {% endif %}",
      }),
    );

    const template = await loadChatTemplate(directory);
    expect(template).not.toBeNull();
    expect(
      template?.format(
        [
          { role: "user", content: "Hello" },
          { role: "assistant", content: "Hi" },
        ],
        { addGenerationPrompt: true },
      ),
    ).toBe("<s>[user] Hello[assistant] Hi[assistant] ");
  });

  test("prefers chat_template.jinja over tokenizer_config.json", async () => {
    const directory = createTempDir("mlxts-transformers-chat-template-file-");
    writeFileSync(join(directory, "config.json"), JSON.stringify({ model_type: "llama" }));
    writeFileSync(
      join(directory, "tokenizer_config.json"),
      JSON.stringify({
        bos_token: "<s>",
        eos_token: "</s>",
        chat_template: "{{ bos_token }}wrong",
      }),
    );
    writeFileSync(
      join(directory, "chat_template.jinja"),
      "{{ bos_token }}{{ messages[0]['content'] }}",
    );

    const template = await loadChatTemplate(directory);
    expect(template?.format([{ role: "user", content: "Hello" }])).toBe("<s>Hello");
  });

  test("returns null when no template is present", async () => {
    const directory = createTempDir("mlxts-transformers-chat-template-none-");
    writeFileSync(join(directory, "config.json"), JSON.stringify({ model_type: "llama" }));

    expect(await loadChatTemplate(directory)).toBeNull();
  });

  test("loads array-valued templates from tokenizer_config and uses special token maps", async () => {
    const directory = createTempDir("mlxts-transformers-chat-template-array-");
    writeFileSync(join(directory, "config.json"), JSON.stringify({ model_type: "llama" }));
    writeFileSync(
      join(directory, "tokenizer_config.json"),
      JSON.stringify({
        chat_template: [
          { name: "tool", template: "ignored" },
          {
            name: "default",
            template:
              "{{ bos_token }}{% for message in messages %}[{{ message['role'] }}] {{ message['content'] }}{% endfor %}{{ eos_token }}",
          },
        ],
      }),
    );
    writeFileSync(
      join(directory, "special_tokens_map.json"),
      JSON.stringify({
        bos_token: { content: "<bos>" },
        eos_token: { content: "<eos>" },
      }),
    );

    const template = await loadChatTemplate(directory);
    expect(
      template?.format(
        [
          { role: "system", content: "Rules" },
          { role: "user", content: "Hello" },
        ],
        { addGenerationPrompt: false },
      ),
    ).toBe("<bos>[system] Rules[user] Hello<eos>");
  });

  test("falls back to processor and config chat-template fields when files are absent or empty", async () => {
    const processorDirectory = createTempDir("mlxts-transformers-chat-template-processor-");
    writeFileSync(join(processorDirectory, "config.json"), JSON.stringify({ model_type: "llama" }));
    writeFileSync(
      join(processorDirectory, "processor_config.json"),
      JSON.stringify({
        chat_template:
          "{{ bos_token }}{% for message in messages %}{{ message['content'] }}{% endfor %}",
      }),
    );
    writeFileSync(
      join(processorDirectory, "special_tokens_map.json"),
      JSON.stringify({ bos_token: "<proc>" }),
    );

    const fromProcessor = await loadChatTemplate(processorDirectory);
    expect(fromProcessor?.format([{ role: "user", content: "Hello" }])).toBe("<proc>Hello");

    const configDirectory = createTempDir("mlxts-transformers-chat-template-config-");
    writeFileSync(
      join(configDirectory, "config.json"),
      JSON.stringify({
        model_type: "llama",
        chat_template_jinja:
          "{{ bos_token }}{% for message in messages %}{{ message['content'] }}{% endfor %}",
      }),
    );
    writeFileSync(join(configDirectory, "chat_template.jinja"), "   \n");
    writeFileSync(
      join(configDirectory, "special_tokens_map.json"),
      JSON.stringify({ bos_token: "<cfg>" }),
    );

    const fromConfig = await loadChatTemplate(configDirectory);
    expect(fromConfig?.format([{ role: "user", content: "Hi" }])).toBe("<cfg>Hi");
  });

  test("renders OpenAI wire-format tool call arguments as template mappings", async () => {
    const directory = createTempDir("mlxts-transformers-chat-template-tools-");
    writeFileSync(join(directory, "config.json"), JSON.stringify({ model_type: "qwen3_5" }));
    writeFileSync(
      join(directory, "chat_template.jinja"),
      [
        "{% for message in messages %}",
        "{% if message.role == 'assistant' %}",
        "{% for tool_call in message.tool_calls %}",
        "{% set fn = tool_call.function %}",
        "{% for name, value in fn.arguments|items %}{{ name }}={{ value }};{% endfor %}",
        "{% endfor %}",
        "{% endif %}",
        "{% endfor %}",
      ].join(""),
    );

    const template = await loadChatTemplate(directory);
    expect(
      template?.format(
        [
          {
            role: "assistant",
            content: "",
            tool_calls: [
              {
                id: "call-read",
                type: "function",
                function: { name: "read", arguments: '{"path":"package.json"}' },
              },
            ],
          },
        ],
        { addGenerationPrompt: false },
      ),
    ).toBe("path=package.json;");
  });

  test("preserves Gemma empty thinking channels in assistant history when requested", async () => {
    const directory = createTempDir("mlxts-transformers-chat-template-gemma-thinking-");
    writeFileSync(join(directory, "config.json"), JSON.stringify({ model_type: "gemma4" }));
    writeFileSync(
      join(directory, "chat_template.jinja"),
      [
        "{{ bos_token }}",
        "{% for message in messages %}",
        "{{ '<|turn>' + ('model' if message.role == 'assistant' else message.role) + '\\n' }}",
        "{{ message.content }}",
        "{{ '<turn|>\\n' }}",
        "{% endfor %}",
        "{% if add_generation_prompt %}",
        "{{ '<|turn>model\\n' }}",
        "{% if not enable_thinking | default(false) %}",
        "{{ '<|channel>thought\\n<channel|>' }}",
        "{% endif %}",
        "{% endif %}",
      ].join(""),
    );
    writeFileSync(join(directory, "tokenizer_config.json"), JSON.stringify({ bos_token: "<bos>" }));

    const template = await loadChatTemplate(directory);
    expect(
      template?.format(
        [
          { role: "user", content: "Hi" },
          { role: "assistant", content: "Hello" },
          { role: "user", content: "Again" },
        ],
        { enableThinking: false, preserveThinking: true },
      ),
    ).toBe(
      "<bos><|turn>user\nHi<turn|>\n<|turn>model\n<|channel>thought\n<channel|>Hello<turn|>\n<|turn>user\nAgain<turn|>\n<|turn>model\n<|channel>thought\n<channel|>",
    );
  });

  test("preserves Gemma empty thinking channels before assistant tool-call history", async () => {
    const directory = createTempDir("mlxts-transformers-chat-template-gemma-tool-thinking-");
    writeFileSync(join(directory, "config.json"), JSON.stringify({ model_type: "gemma4" }));
    writeFileSync(
      join(directory, "chat_template.jinja"),
      [
        "{{ bos_token }}",
        "{% for message in messages %}",
        "{% if message.role == 'assistant' %}",
        "{{ '<|turn>model\\n' }}",
        "{% for tool_call in message.tool_calls %}",
        "{{ '<|tool_call>call:' + tool_call.function.name + '{}<tool_call|>' }}",
        "{% endfor %}",
        "{% else %}",
        "{{ '<|turn>' + message.role + '\\n' + message.content + '<turn|>\\n' }}",
        "{% endif %}",
        "{% endfor %}",
      ].join(""),
    );
    writeFileSync(join(directory, "tokenizer_config.json"), JSON.stringify({ bos_token: "<bos>" }));

    const template = await loadChatTemplate(directory);
    expect(
      template?.format(
        [
          { role: "user", content: "Read" },
          {
            role: "assistant",
            content: "",
            tool_calls: [
              {
                id: "call-read",
                type: "function",
                function: { name: "read", arguments: "{}" },
              },
            ],
          },
        ],
        { addGenerationPrompt: false, enableThinking: false, preserveThinking: true },
      ),
    ).toBe(
      "<bos><|turn>user\nRead<turn|>\n<|turn>model\n<|channel>thought\n<channel|><|tool_call>call:read{}<tool_call|>",
    );
  });

  test("renders Gemma-style reasoning, tool calls, and tool responses", async () => {
    const directory = createTempDir("mlxts-transformers-chat-template-gemma-tools-");
    writeFileSync(join(directory, "config.json"), JSON.stringify({ model_type: "gemma4" }));
    writeFileSync(
      join(directory, "chat_template.jinja"),
      [
        "{{ bos_token }}",
        "{% if tools %}",
        "<|turn>system\n",
        "{% for tool in tools %}",
        '<|tool>declaration:{{ tool.function.name }}{description:<|"|>{{ tool.function.description }}<|"|>}<tool|>',
        "{% endfor %}",
        "<turn|>\n",
        "{% endif %}",
        "{% set ns = namespace(prev_message_type=None) %}",
        "{% for message in messages %}",
        "{% if message.role != 'tool' %}",
        "{% set role = 'model' if message.role == 'assistant' else message.role %}",
        "{{ '<|turn>' + role + '\\n' }}",
        "{% if message.reasoning_content and message.tool_calls %}",
        "{{ '<|channel>thought\\n' + message.reasoning_content + '\\n<channel|>' }}",
        "{% endif %}",
        "{% if message.tool_calls %}",
        "{% for tool_call in message.tool_calls %}",
        "{% set function = tool_call.function %}",
        "{{ '<|tool_call>call:' + function.name + '{' }}",
        "{% for name, value in function.arguments|items %}",
        "{{ name }}:<|\"|>{{ value }}<|\"|>{{ ',' if not loop.last }}",
        "{% endfor %}",
        "{{ '}<tool_call|>' }}",
        "{% endfor %}",
        "{% set ns.prev_message_type = 'tool_call' %}",
        "{% for follow in messages[loop.index:] %}",
        "{% if follow.role == 'tool' %}",
        "{{ '<|tool_response>response:' + message.tool_calls[0].function.name + '{value:<|\"|>' + follow.content + '<|\"|>}<tool_response|>' }}",
        "{% set ns.prev_message_type = 'tool_response' %}",
        "{% endif %}",
        "{% endfor %}",
        "{% endif %}",
        "{{ message.content }}",
        "{% if ns.prev_message_type != 'tool_response' %}<turn|>\n{% endif %}",
        "{% endif %}",
        "{% endfor %}",
        "{% if add_generation_prompt and ns.prev_message_type != 'tool_response' %}<|turn>model\n{% endif %}",
      ].join(""),
    );
    writeFileSync(join(directory, "tokenizer_config.json"), JSON.stringify({ bos_token: "<bos>" }));

    const template = await loadChatTemplate(directory);
    const rendered = template?.format(
      [
        { role: "user", content: "Read package.json" },
        {
          role: "assistant",
          content: "",
          reasoning_content: "Need the file.",
          tool_calls: [
            {
              id: "call-read",
              type: "function",
              function: { name: "read", arguments: '{"path":"package.json"}' },
            },
          ],
        },
        { role: "tool", tool_call_id: "call-read", content: '{"name":"mlxts"}' },
      ],
      {
        addGenerationPrompt: true,
        tools: [{ type: "function", function: { name: "read", description: "Read a file" } }],
      },
    );

    expect(rendered).toContain("<|tool>declaration:read");
    expect(rendered).toContain("<|channel>thought\nNeed the file.\n<channel|>");
    expect(rendered).toContain('<|tool_call>call:read{path:<|"|>package.json<|"|>}<tool_call|>');
    expect(rendered).toContain(
      '<|tool_response>response:read{value:<|"|>{"name":"mlxts"}<|"|>}<tool_response|>',
    );
    expect(rendered?.endsWith("<|turn>model\n")).toBe(false);
  });
});
