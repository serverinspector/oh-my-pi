import { describe, expect, it } from "bun:test";
import { type CompactionSettings, DEFAULT_COMPACTION_SETTINGS } from "@oh-my-pi/pi-agent-core/compaction";
import { resolveAdvisorCompactionSettings } from "@oh-my-pi/pi-coding-agent/session/agent-session";

function settings(strategy: CompactionSettings["strategy"]): CompactionSettings {
	return { ...DEFAULT_COMPACTION_SETTINGS, strategy };
}

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
