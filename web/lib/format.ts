import type { Source } from "@/lib/api";

export function formatSeconds(s: number): string {
  return `${s < 0.1 ? s.toFixed(2) : s.toFixed(1)} s`;
}

export function formatCost(usd: number): string {
  return `$${usd.toFixed(4)}`;
}

export const SOURCE_LABEL: Record<Source, string> = {
  slack: "Slack",
  gmail: "Gmail",
  google_drive: "Google Drive",
  confluence: "Confluence",
  jira: "Jira",
  linear: "Linear",
  github: "GitHub",
  hubspot: "HubSpot",
  fireflies: "Fireflies",
};

// Solid badge colours, one per source, each with white text at a contrast of 4.5:1 or better in both themes.
export const SOURCE_COLOR: Record<Source, string> = {
  slack: "#7e22ce",
  gmail: "#c62828",
  google_drive: "#15803d",
  confluence: "#1d4ed8",
  jira: "#0e7490",
  linear: "#8a5a00",
  github: "#4b5563",
  hubspot: "#c2410c",
  fireflies: "#be185d",
};

export const VERDICT_LABEL = {
  supported: "Supported",
  partially_supported: "Partially supported",
  unsupported: "Unsupported",
  contradicted: "Contradicted",
} as const;
