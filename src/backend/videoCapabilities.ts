import type {
  VideoAdapter,
  VideoCapabilities,
  VideoInference,
} from "./rules.ts";
import { tNow } from "../i18n/messages.ts";

export interface VideoOutputDraft {
  resolution: string;
  seconds: string;
  width: string;
  height: string;
}

export interface VideoInferenceDraft {
  steps: string;
  flowShift: string;
  audioFlowShift: string;
  guidanceScale: string;
  loraName: string;
  loraPath: string;
  loraScale: string;
  regenerate?: boolean;
}

/** Distilled adapters carry their step count in the file name, e.g. 8step_v1.0_768p. */
const STEPS_IN_NAME = /(\d+)[-_ ]step/i;

export function parseVideoSeconds(value: string): number[] {
  const result = new Set<number>();
  for (const token of value.trim().split(/[,，\s]+/)) {
    const match = /^(\d+)(?:-(\d+))?$/.exec(token);
    if (!match)
      throw new Error(tNow("errors.videoCapabilities.durationFormat"));
    const start = Number(match[1]),
      end = Number(match[2] ?? match[1]);
    if (start < 1 || end > 60 || end < start)
      throw new Error(tNow("errors.videoCapabilities.durationRange"));
    for (let second = start; second <= end; second++) result.add(second);
  }
  return [...result].sort((a, b) => a - b);
}

export function videoOutputDrafts(
  caps: VideoCapabilities | null | undefined,
): VideoOutputDraft[] {
  return (
    caps?.outputs.map((item) => ({
      resolution: item.resolution,
      seconds: item.durations.join(","),
      width: String(item.width ?? ""),
      height: String(item.height ?? ""),
    })) ?? [{ resolution: "", seconds: "", width: "", height: "" }]
  );
}

export function createVideoInference(
  draft: VideoInferenceDraft,
): VideoInference | undefined {
  if (draft.regenerate) {
    // A regenerate build fixes its own schedule server-side and refuses an
    // adapter or a step count, so nothing else may travel with the flag.
    if (
      [draft.steps, draft.flowShift, draft.audioFlowShift, draft.guidanceScale,
        draft.loraName, draft.loraPath, draft.loraScale].some((value) =>
        value.trim(),
      )
    )
      throw new Error(tNow("errors.videoCapabilities.regenerateOwnsSchedule"));
    return { regenerate: true };
  }
  const loraName = draft.loraName.trim();
  const loraPath = draft.loraPath.trim();
  const numbers = [
    draft.steps,
    draft.flowShift,
    draft.audioFlowShift,
    draft.guidanceScale,
    draft.loraScale,
  ].map((value) => value.trim());
  if (!numbers.some(Boolean) && !loraName && !loraPath) return undefined;
  const read = (value: string, min: number, max: number, integer = false) => {
    if (!value.trim()) return undefined;
    const parsed = Number(value);
    if (
      !Number.isFinite(parsed) ||
      parsed < min ||
      parsed > max ||
      (integer && !Number.isInteger(parsed))
    )
      throw new Error(tNow("errors.videoCapabilities.inferenceRange"));
    return parsed;
  };
  if (Boolean(loraName) !== Boolean(loraPath))
    throw new Error(tNow("errors.videoCapabilities.loraPair"));
  const steps = read(draft.steps, 1, 50, true);
  const flowShift = read(draft.flowShift, 0, 60);
  const audioFlowShift = read(draft.audioFlowShift, 0, 60);
  const guidanceScale = read(draft.guidanceScale, 0, 20);
  const inference: VideoInference = {};
  if (steps !== undefined) inference.steps = steps;
  if (flowShift !== undefined) inference.flowShift = flowShift;
  if (audioFlowShift !== undefined) inference.audioFlowShift = audioFlowShift;
  if (guidanceScale !== undefined) inference.guidanceScale = guidanceScale;
  if (loraName) {
    const scale = read(draft.loraScale, 0, 1);
    const derived = STEPS_IN_NAME.exec(loraName);
    const distilled = derived ? Number(derived[1]) : undefined;
    if (
      steps !== undefined &&
      distilled !== undefined &&
      steps !== distilled &&
      steps !== distilled + 1
    )
      throw new Error(tNow("errors.videoCapabilities.loraSteps"));
    inference.lora = scale === undefined
      ? { name: loraName, path: loraPath }
      : { name: loraName, path: loraPath, scale };
  }
  return Object.keys(inference).length ? inference : undefined;
}

export function createVideoCapabilities(
  rows: VideoOutputDraft[],
  defaultResolution: string,
  defaultDuration: number,
  dimensionStep: VideoCapabilities["dimensionStep"],
  adapter: VideoAdapter = "auto",
  inference?: VideoInference,
): VideoCapabilities {
  if (!rows.length || rows.length > 20)
    throw new Error(tNow("errors.videoCapabilities.outputCount"));
  const outputs = rows.map((row) => {
    const resolution = row.resolution.trim().toUpperCase();
    if (
      !/^[A-Z0-9][A-Z0-9_.-]{0,23}$/.test(resolution) ||
      resolution === "AUTO"
    )
      throw new Error(tNow("errors.videoCapabilities.resolutionName"));
    const durations = parseVideoSeconds(row.seconds);
    if (Boolean(row.width.trim()) !== Boolean(row.height.trim()))
      throw new Error(tNow("errors.videoCapabilities.pixelPair"));
    if (!row.width.trim()) {
      // The /videos resolver needs pixels except for the two H3 buckets it
      // derives itself; saving anything else would only fail at generation.
      const derived =
        (adapter === "fasth3" || adapter === "minimax_h3") &&
        ["480P", "768P", "1080P"].includes(resolution);
      if (!derived)
        throw new Error(
          tNow("errors.videoCapabilities.pixelRequired"),
        );
      return { resolution, durations };
    }
    const width = Number(row.width),
      height = Number(row.height);
    if (
      ![width, height].every(
        (edge) =>
          Number.isInteger(edge) &&
          edge >= 64 &&
          edge <= 8192 &&
          edge % dimensionStep === 0,
      ) ||
      width < height
    )
      throw new Error(
        tNow("errors.videoCapabilities.pixelBounds"),
      );
    return { resolution, durations, width, height };
  });
  if (new Set(outputs.map((item) => item.resolution)).size !== outputs.length)
    throw new Error(tNow("errors.videoCapabilities.duplicateResolution"));
  if (
    !outputs.some(
      (item) =>
        (defaultResolution === "auto" ||
          item.resolution === defaultResolution) &&
        item.durations.includes(defaultDuration),
    )
  )
    throw new Error(tNow("errors.videoCapabilities.defaultOutput"));
  return {
    outputs,
    defaultResolution,
    defaultDuration,
    dimensionStep,
    ...(inference ? { inference } : {}),
  };
}
