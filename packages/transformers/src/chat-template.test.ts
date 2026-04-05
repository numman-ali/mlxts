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
});
