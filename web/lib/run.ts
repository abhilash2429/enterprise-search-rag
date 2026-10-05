// State of one /ask run as its events arrive, and the per-stage view the timeline renders.

import type { AskEvent, AskEventMap, AskOutcome, Stage } from "@/lib/api";

export const DEFAULT_STAGES: Stage[] = ["route", "retrieve", "fuse", "rerank", "generate", "verify"];

export type RunState = {
  id: number;
  question: string;
  phase: "idle" | "checking" | "streaming" | "finished";
  startedAt: number; // client clock (ms) when the run began
  events: Partial<AskEventMap>;
  arrivals: Partial<Record<keyof AskEventMap, number>>; // client clock (ms) per event
  outcome: AskOutcome | null;
  finishedAt: number | null;
};

export type RunAction =
  | { type: "begin"; id: number; question: string; at: number }
  | { type: "event"; id: number; event: AskEvent; at: number }
  | { type: "finish"; id: number; outcome: AskOutcome; at: number }
  | { type: "reset" };

export const IDLE: RunState = {
  id: 0,
  question: "",
  phase: "idle",
  startedAt: 0,
  events: {},
  arrivals: {},
  outcome: null,
  finishedAt: null,
};

export function runReducer(state: RunState, action: RunAction): RunState {
  if (action.type === "reset") return { ...IDLE, id: state.id };
  if (action.type === "begin") {
    return { ...IDLE, id: action.id, question: action.question, phase: "checking", startedAt: action.at };
  }
  if (action.id !== state.id || state.phase === "finished") return state;
  if (action.type === "event") {
    return {
      ...state,
      phase: "streaming",
      events: { ...state.events, [action.event.event]: action.event.data },
      arrivals: { ...state.arrivals, [action.event.event]: action.at },
    };
  }
  return { ...state, phase: "finished", outcome: action.outcome, finishedAt: action.at };
}

export type StageStatus = "pending" | "running" | "done" | "failed" | "stopped";
export type StageView = { stage: Stage; status: StageStatus; seconds: number | null; since: number | null };

/** One entry per stage in start.stages (the default six before `start` arrives). */
export function stageViews(state: RunState): StageView[] {
  const stages = state.events.start?.stages ?? DEFAULT_STAGES;
  const firstOpen = stages.findIndex((s) => state.events[s] === undefined);
  return stages.map((stage, i) => {
    const data = state.events[stage];
    if (data !== undefined) return { stage, status: "done", seconds: data.seconds, since: null };
    if (i !== firstOpen || !state.events.start) return { stage, status: "pending", seconds: null, since: null };
    const prev = i === 0 ? "start" : stages[i - 1];
    const since = state.arrivals[prev] ?? state.startedAt;
    if (state.phase === "streaming") return { stage, status: "running", seconds: null, since };
    if (state.outcome?.kind === "aborted") return { stage, status: "stopped", seconds: null, since: null };
    if (state.outcome && state.outcome.kind !== "done") return { stage, status: "failed", seconds: null, since: null };
    return { stage, status: "pending", seconds: null, since: null };
  });
}

/** The stage currently running, if any. */
export function runningStage(state: RunState): Stage | null {
  return stageViews(state).find((v) => v.status === "running")?.stage ?? null;
}

/** A human-readable reason when the run ended without `done`. */
export function failureMessage(outcome: AskOutcome | null): string | null {
  if (!outcome || outcome.kind === "done") return null;
  if (outcome.kind === "aborted") return "Stopped.";
  return outcome.message;
}
