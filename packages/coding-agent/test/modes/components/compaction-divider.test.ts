/**
 * Contract: the compaction point renders as a slim horizontal divider —
 * `── 📷 compacted · ctrl+o ──` — instead of a full summary box, keeping the
 * transcript visually continuous. Expansion (ctrl+o) reveals the summary.
 * The render cache must honor the pi-tui same-reference contract: unchanged
 * components return the identical array so containers can memoize.
 */

import { beforeAll, describe, expect, it } from "bun:test";
import { createCompactionSummaryMessage } from "@oh-my-pi/pi-agent-core/compaction";
import type { ImageContent } from "@oh-my-pi/pi-ai";
import { CompactionSummaryMessageComponent } from "@oh-my-pi/pi-coding-agent/modes/components/compaction-summary-message";
import { initTheme } from "@oh-my-pi/pi-coding-agent/modes/theme/theme";
import { buildSessionContext } from "@oh-my-pi/pi-coding-agent/session/session-context";
import type { CompactionEntry, SessionEntry } from "@oh-my-pi/pi-coding-agent/session/session-entries";

beforeAll(() => {
	initTheme();
});

const SUMMARY = "Earlier the user fixed the login TTL bug.";

function makeComponent(images?: ImageContent[]): CompactionSummaryMessageComponent {
	return new CompactionSummaryMessageComponent(
		createCompactionSummaryMessage(SUMMARY, 84000, new Date().toISOString(), undefined, undefined, images),
	);
}

describe("CompactionSummaryMessageComponent", () => {
	it("collapsed: a single full-width divider carrying the expand affordance", () => {
		const lines = makeComponent().render(80);
		expect(lines.length).toBe(3); // breathing room above and below the rule
		const rule = Bun.stripANSI(lines[1]);
		expect(rule).toContain("compacted");
		expect(rule).toContain("ctrl+o");
		// The rule spans the full width and hides the summary body.
		expect(Bun.stringWidth(rule)).toBe(80);
		expect(rule).not.toContain(SUMMARY);
	});

	it("expanded: reveals the summary (and snapcompact frame count) below the divider", () => {
		const component = makeComponent([{ type: "image", data: "ZmFrZQ==", mimeType: "image/png" }]);
		component.setExpanded(true);
		const text = Bun.stripANSI(component.render(80).join("\n"));
		expect(text).toContain("compacted");
		expect(text).toContain(SUMMARY);
		expect(text).toContain("tokens");
		expect(text).toContain("1 snapcompact frame attached");
	});

	it("collapsed: names Codex V2 native compactions in the transcript", () => {
		const component = new CompactionSummaryMessageComponent(
			createCompactionSummaryMessage(
				"Context compacted by Codex V2 remote compaction. Provider-native replacement history carries the retained context.",
				128000,
				new Date().toISOString(),
				"Codex V2 remote compaction",
			),
		);
		const collapsed = Bun.stripANSI(component.render(80).join("\n"));
		expect(collapsed).toContain("Codex V2 compacted");
		expect(collapsed).toContain("ctrl+o");
		expect(collapsed).not.toContain("Provider-native replacement history");

		const narrow = Bun.stripANSI(component.render(32).join("\n"));
		expect(narrow).toContain("Codex V2 compacted");
		expect(narrow).toContain("─");
		expect(narrow).not.toContain("Provider-native replacement history");

		component.setExpanded(true);
		const expanded = Bun.stripANSI(component.render(80).join("\n"));
		expect(expanded).toContain("Codex V2 provider-native compaction item attached");
	});

	it("renders the collapsed transcript marker for a codex-v2 compact entry", () => {
		const timestamp = new Date().toISOString();
		const oldUser: SessionEntry = {
			type: "message",
			id: "old-user",
			parentId: null,
			timestamp,
			message: { role: "user", content: "old context", timestamp: Date.now() },
		};
		const keptUser: SessionEntry = {
			type: "message",
			id: "kept-user",
			parentId: oldUser.id,
			timestamp,
			message: { role: "user", content: "kept context", timestamp: Date.now() },
		};
		const compaction: CompactionEntry = {
			type: "compaction",
			id: "codex-v2-compact",
			parentId: keptUser.id,
			timestamp,
			summary:
				"Context compacted by Codex V2 remote compaction. Provider-native replacement history carries the retained context.",
			shortSummary: "Codex V2 remote compaction",
			firstKeptEntryId: keptUser.id,
			tokensBefore: 128000,
			preserveData: {
				openaiRemoteCompaction: {
					provider: "openai-codex",
					method: "codex-v2",
					replacementHistory: [
						{ type: "message", role: "user", content: [{ type: "input_text", text: "kept context" }] },
						{ type: "compaction", encrypted_content: "encrypted" },
					],
					compactionItem: { type: "compaction", encrypted_content: "encrypted" },
				},
			},
		};

		const transcript = buildSessionContext([oldUser, keptUser, compaction], undefined, undefined, {
			transcript: true,
			collapseCompactedHistory: true,
		});

		expect(transcript.messages.map(message => message.role)).toEqual(["compactionSummary", "user"]);
		const summaryMessage = transcript.messages[0];
		if (summaryMessage?.role !== "compactionSummary") {
			throw new Error("collapsed transcript did not expose the compaction marker");
		}
		const rendered = Bun.stripANSI(new CompactionSummaryMessageComponent(summaryMessage).render(32).join("\n"));
		expect(rendered).toContain("Codex V2 compacted");
		expect(rendered).toContain("─");
	});

	it("degrades to a bare label when the viewport is too narrow for a framed rule", () => {
		const lines = makeComponent().render(10);
		expect(Bun.stripANSI(lines[1])).toContain("compacted");
	});

	it("honors the same-reference render cache and busts it on expansion toggle", () => {
		const component = makeComponent();
		const first = component.render(80);
		expect(component.render(80)).toBe(first);
		component.setExpanded(true);
		const expanded = component.render(80);
		expect(expanded).not.toBe(first);
		expect(component.render(80)).toBe(expanded);
	});
});
