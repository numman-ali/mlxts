import { afterEach, describe, expect, test } from "bun:test";
import { mkdtempSync, rmSync, writeFileSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";
import { loadInteractionProfile } from "./interaction-profile";
import { loadPretrainedTokenizer } from "./load";

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

function writeSnapshot(
  directory: string,
  modelType: string,
  tokenizerJson: Record<string, unknown>,
  tokenizerConfig: Record<string, unknown>,
  specialTokensMap: Record<string, unknown> = {},
): void {
  writeFileSync(join(directory, "config.json"), JSON.stringify({ model_type: modelType }));
  writeFileSync(join(directory, "tokenizer.json"), JSON.stringify(tokenizerJson));
  writeFileSync(join(directory, "tokenizer_config.json"), JSON.stringify(tokenizerConfig));
  writeFileSync(join(directory, "special_tokens_map.json"), JSON.stringify(specialTokensMap));
}

function byteLevelTokenizerJson(
  templateTokens: Array<{ id: number; content: string }>,
  vocab: Record<string, number>,
): Record<string, unknown> {
  return {
    model: {
      type: "BPE",
      vocab,
      merges: [],
      unk_token: "<eos>",
      byte_fallback: false,
    },
    added_tokens: [
      { id: 0, content: "<eos>", special: true },
      { id: 1, content: "<s>", special: true },
      ...templateTokens.map((token) => ({ ...token, special: true })),
    ],
    pre_tokenizer: {
      type: "ByteLevel",
      add_prefix_space: false,
      trim_offsets: true,
      use_regex: true,
    },
    decoder: {
      type: "ByteLevel",
    },
  };
}

function instTokenizerJson(): Record<string, unknown> {
  return byteLevelTokenizerJson(
    [
      { id: 2, content: "[INST]" },
      { id: 3, content: "[/INST]" },
    ],
    {
      "<eos>": 0,
      "<s>": 1,
      H: 10,
      e: 11,
      l: 12,
      o: 13,
    },
  );
}

function phiTokenizerJson(): Record<string, unknown> {
  return byteLevelTokenizerJson(
    [
      { id: 2, content: "<|user|>" },
      { id: 3, content: "<|assistant|>" },
    ],
    {
      "<eos>": 0,
      "<s>": 1,
      H: 10,
      e: 11,
      l: 12,
      o: 13,
      Ċ: 14,
    },
  );
}

function gemmaTokenizerJson(): Record<string, unknown> {
  return {
    model: {
      type: "BPE",
      vocab: {
        "<pad>": 0,
        "<eos>": 1,
        "<bos>": 2,
        "<unk>": 3,
        "\n": 107,
        user: 2364,
        model: 4368,
        Hello: 9259,
        "▁there": 993,
      },
      merges: [],
      unk_token: "<unk>",
      byte_fallback: false,
    },
    added_tokens: [
      { id: 0, content: "<pad>", special: true },
      { id: 1, content: "<eos>", special: true },
      { id: 2, content: "<bos>", special: true },
      { id: 3, content: "<unk>", special: true },
      { id: 105, content: "<|turn>", special: true },
      { id: 106, content: "<turn|>", special: true },
    ],
    pre_tokenizer: {
      type: "Split",
      pattern: { String: " " },
      behavior: "MergedWithPrevious",
      invert: false,
    },
    decoder: {
      type: "Sequence",
      decoders: [
        { type: "Replace", pattern: { String: "▁" }, content: " " },
        { type: "ByteFallback" },
        { type: "Fuse" },
      ],
    },
  };
}

describe("interaction profiles", () => {
  test("compile chat prompts with token parity for llama and mistral style templates", async () => {
    for (const modelType of ["llama", "mistral"]) {
      const directory = createTempDir(`mlxts-interaction-${modelType}-`);
      writeSnapshot(
        directory,
        modelType,
        instTokenizerJson(),
        {
          bos_token: "<s>",
          eos_token: "<eos>",
          add_bos_token: false,
          add_eos_token: false,
          chat_template:
            "{{ bos_token }}[INST]{% for message in messages %}{{ message['content'] }}{% endfor %}[/INST]",
        },
        {
          bos_token: { content: "<s>" },
          eos_token: { content: "<eos>" },
        },
      );

      const tokenizer = await loadPretrainedTokenizer(directory);
      const profile = await loadInteractionProfile(directory);
      const compiled = profile.compileMessages(tokenizer, [{ role: "user", content: "Hello" }], {
        addGenerationPrompt: true,
      });

      expect(compiled.text).toBe("<s>[INST]Hello[/INST]");
      expect(compiled.tokenIds).toEqual([1, 2, 10, 11, 12, 12, 13, 3]);
    }
  });

  test("compile chat prompts with token parity for phi style templates", async () => {
    const directory = createTempDir("mlxts-interaction-phi-");
    writeSnapshot(
      directory,
      "phi3",
      phiTokenizerJson(),
      {
        bos_token: "<s>",
        eos_token: "<eos>",
        add_bos_token: false,
        add_eos_token: false,
        chat_template: "<|user|>\n{{ messages[0]['content'] }}\n<|assistant|>\n",
      },
      {
        bos_token: { content: "<s>" },
        eos_token: { content: "<eos>" },
      },
    );

    const tokenizer = await loadPretrainedTokenizer(directory);
    const profile = await loadInteractionProfile(directory);
    const compiled = profile.compileMessages(tokenizer, [{ role: "user", content: "Hello" }], {
      addGenerationPrompt: true,
    });

    expect(compiled.text).toBe("<|user|>\nHello\n<|assistant|>");
    expect(compiled.tokenIds).toEqual([2, 14, 10, 11, 12, 12, 13, 14, 3]);
  });

  test("compile chat prompts with token parity for gemma turn markers", async () => {
    const directory = createTempDir("mlxts-interaction-gemma-");
    writeSnapshot(
      directory,
      "gemma4_text",
      gemmaTokenizerJson(),
      {
        bos_token: "<bos>",
        eos_token: "<eos>",
        pad_token: "<pad>",
        unk_token: "<unk>",
        add_bos_token: false,
        add_eos_token: false,
        chat_template:
          "{{ bos_token }}{% for message in messages %}<|turn>{{ message['role'] }}\n{{ message['content'] }}<turn|>\n{% endfor %}{% if add_generation_prompt %}<|turn>model\n{% endif %}",
      },
      {
        bos_token: { content: "<bos>" },
        eos_token: { content: "<eos>" },
        pad_token: { content: "<pad>" },
        unk_token: { content: "<unk>" },
      },
    );

    const tokenizer = await loadPretrainedTokenizer(directory);
    const profile = await loadInteractionProfile(directory);
    const compiled = profile.compileMessages(
      tokenizer,
      [{ role: "user", content: "Hello there" }],
      {
        addGenerationPrompt: true,
      },
    );

    expect(compiled.text).toBe("<bos><|turn>user\nHello there<turn|>\n<|turn>model\n");
    expect(compiled.tokenIds).toEqual([2, 105, 2364, 107, 9259, 993, 106, 107, 105, 4368, 107]);
  });

  test("passes thinking controls through Qwen-style chat templates", async () => {
    const directory = createTempDir("mlxts-interaction-qwen-thinking-");
    writeSnapshot(directory, "qwen3_5_text", phiTokenizerJson(), {
      add_bos_token: false,
      add_eos_token: false,
      chat_template:
        "{% for message in messages %}{{ message['role'] + ':' + message['content'] + '\\n' }}{% endfor %}{% if add_generation_prompt %}assistant:\n{% if enable_thinking is defined and enable_thinking is false %}<think>\n\n</think>\n\n{% else %}<think>\n{% endif %}{% endif %}",
    });

    const tokenizer = await loadPretrainedTokenizer(directory);
    const profile = await loadInteractionProfile(directory);
    const thinking = profile.compileMessages(tokenizer, [{ role: "user", content: "Hello" }], {
      addGenerationPrompt: true,
    });
    const noThinking = profile.compileMessages(tokenizer, [{ role: "user", content: "Hello" }], {
      addGenerationPrompt: true,
      enableThinking: false,
    });

    expect(thinking.text).toContain("assistant:\n<think>\n");
    expect(noThinking.text).toContain("assistant:\n<think>\n\n</think>\n\n");
  });

  test("reject direct message compilation when no chat template exists", async () => {
    const directory = createTempDir("mlxts-interaction-completion-");
    writeSnapshot(
      directory,
      "llama",
      instTokenizerJson(),
      {
        bos_token: "<s>",
        eos_token: "<eos>",
        add_bos_token: false,
        add_eos_token: false,
      },
      {
        bos_token: { content: "<s>" },
        eos_token: { content: "<eos>" },
      },
    );

    const tokenizer = await loadPretrainedTokenizer(directory);
    const profile = await loadInteractionProfile(directory);

    expect(profile.kind).toBe("completion");
    expect(() => profile.compileMessages(tokenizer, [{ role: "user", content: "Hello" }])).toThrow(
      "interaction profile: this checkpoint does not define a chat template",
    );
  });
});
