import { jsonFileResponse, readFixtureText } from "@/lib/mock";

export async function GET(): Promise<Response> {
  return jsonFileResponse(await readFixtureText("questions.json"));
}
