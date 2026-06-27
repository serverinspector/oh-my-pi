import { describe, expect, it } from "bun:test";
import { type CompactionSettings, DEFAULT_COMPACTION_SETTINGS } from "@oh-my-pi/pi-agent-core/compaction";
import { buildModel } from "@oh-my-pi/pi-catalog/build";
import type { ModelSpec } from "@oh-my-pi/pi-catalog/types";
import {
	resolveAdvisorCompactionSettings,
	resolveCodexV2ActiveCompactionCandidates,
} from "@oh-my-pi/pi-coding-agent/session/agent-session";

function settings(strategy: CompactionSettings["strategy"]): CompactionSettings {
	return { ...DEFAULT_COMPACTION_SETTINGS, strategy };
}

function openAiModel(overrides: Partial<ModelSpec<"openai-responses">> = {}) {
	return buildModel({
		id: "gpt-5",
		name: "GPT-5",
		api: "openai-responses",
		provider: "openai",
		baseUrl: "https://api.openai.com/v1",
		reasoning: true,
		input: ["text"],
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		contextWindow: 400000,
		maxTokens: 128000,
		...overrides,
	});
}

function anthropicModel(overrides: Partial<ModelSpec<"anthropic">> = {}) {
	return buildModel({
		id: "claude-sonnet-4-5",
		name: "Claude Sonnet 4.5",
		api: "anthropic",
		provider: "anthropic",
		baseUrl: "https://api.anthropic.com/v1",
		reasoning: true,
		input: ["text"],
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		contextWindow: 200000,
		maxTokens: 64000,
		...overrides,
	});
}

describe("resolveCodexV2ActiveCompactionCandidates", () => {
	it("uses only the active model when it can replay Codex V2 history", () => {
		const model = openAiModel();
		expect(resolveCodexV2ActiveCompactionCandidates(model)).toEqual([model]);
	});

	it("rejects non-OpenAI active models instead of relying on fallback candidates", () => {
		expect(resolveCodexV2ActiveCompactionCandidates(anthropicModel())).toEqual([]);
	});
});

describe("resolveAdvisorCompactionSettings", () => {
	it("falls back from codex-v2 to context-full without changing thresholds", () => {
		const input = {
			...settings("codex-v2"),
			thresholdPercent: 86,
			thresholdTokens: 12345,
			keepRecentTokens: 6789,
			reserveTokens: 4321,
		};

		expect(resolveAdvisorCompactionSettings(input)).toEqual({ ...input, strategy: "context-full" });
		expect(input.strategy).toBe("codex-v2");
	});

	it("leaves non-codex-v2 strategies unchanged", () => {
		for (const strategy of ["context-full", "snapcompact", "handoff", "shake", "off"] as const) {
			const input = settings(strategy);
			expect(resolveAdvisorCompactionSettings(input)).toBe(input);
		}
	});
});
