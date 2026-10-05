import { readFile } from "node:fs/promises";

import { documentFile, errorResponse, jsonFileResponse } from "@/lib/mock";

export async function GET(_request: Request, ctx: { params: Promise<{ doc_id: string }> }): Promise<Response> {
  const { doc_id } = await ctx.params;
  const file = documentFile(doc_id);
  // The fixtures hold every doc in a recorded rerank.hits; fuse-only candidates have no document file.
  if (file === null) return errorResponse(404, `unknown doc_id ${doc_id}`);
  return jsonFileResponse(await readFile(file, "utf8"));
}
