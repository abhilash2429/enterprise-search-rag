import {
  errorResponse,
  findRecordedQuestionId,
  mockSpeed,
  readRecordedRun,
  replay,
  sseResponse,
  UNKNOWN_QUESTION_MESSAGE,
} from "@/lib/mock";

export const dynamic = "force-dynamic";

export async function GET(request: Request): Promise<Response> {
  const q = new URL(request.url).searchParams.get("q")?.trim() ?? "";
  if (!q) return errorResponse(400, "missing q");

  const questionId = await findRecordedQuestionId(q);
  const events =
    questionId === null
      ? [{ t: 0, event: "error", data: { message: UNKNOWN_QUESTION_MESSAGE } }]
      : await readRecordedRun(questionId);
  return sseResponse(replay(events, mockSpeed(), request.signal));
}
