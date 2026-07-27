# Long-Session Evidence Packet

Date: 2026-07-27

Status: implemented on `stangoodwin/judge-session-compression`

Representation policy: `complete_session_evidence_weighted_truncate`

Representation version: `2.0`

## Decision

Long traces are converted into one deterministic, extractive, UTF-8-bounded
canonical packet. The implementation does not add an LLM summarization call,
split one trace into multiple judge requests, or drop a trace only because its
content is long.

With full-session embeddings enabled, `SessionEvidencePacketBuilder` builds the
packet before sampling. `SessionEmbeddingCache` embeds that packet in one
provider call, and a kept trace reaches the live evaluator as a
`LiveEvaluationRequest` containing the byte-identical canonical JSON. The
request intentionally has no raw `Trace` field.

If the canonical structure or the minimum mandatory evidence cannot fit,
normalization raises before embedding, sampling, judge submission, or donor
mutation.

## Canonical Schema

```text
{
  policy,
  version,
  session: {
    trace_id,
    agent_id,
    timestamp,
    signature,
    span_count,
    duration_ms,
    status,
    events: [{
      event_index,
      role,
      text,
      tool_name,
      arguments_json,
      output
    }]
  }
}
```

Trace identity, status, event order, roles, and tool names are structural and
are never removed. Only event `text`, canonical `arguments_json`, and `output`
compete for content bytes.

## Allocation

If the complete canonical JSON fits, it is returned unchanged. Otherwise all
variable content is cleared to establish the structural floor. Nonempty
mandatory sources then receive a UTF-8-safe prefix of up to 32 bytes:

| Evidence | Mandatory | Weight |
|---|---:|---:|
| System text | Yes | 6 |
| First user text | Yes | 4 |
| Later user text | Yes | 3 |
| Final assistant text | Yes | 8 |
| Tool-role text and tool output | Yes | 7 |
| Tool arguments | No | 2 |
| Earlier assistant text | No | 2 |

Remaining capacity grows segments in category order: final assistant outcome,
tool results, system context, initial user goal, later user refinements, tool
arguments, and earlier assistant content. Later tool results win ties. Each
growth attempt serializes the complete packet, so JSON escaping is included in
the byte check. Prefix boundaries are backed up to valid UTF-8 code points.

## Audit And Failure Behavior

Every packet records policy, version, original UTF-8 bytes, emitted UTF-8 bytes,
and whether truncation occurred. Unsupported policy/version pairs fail. A
budget below the structural floor raises `max_utf8_bytes too small for
non-content canonical structure`; a budget below the mandatory floor raises
`max_utf8_bytes too small for mandatory task-completion evidence`.

`SESSION_REPRESENTATION_MAX_UTF8_BYTES` configures the packet budget. The
embedding model token limit remains a separate defensive check because UTF-8
bytes are only a conservative token proxy.

## Lineage

Representation policy/version and byte budget participate in the immutable
embedding profile cache version. Semantic-cluster queries and documents carry
that profile as `semantic_scope`; cluster IDs also contain a stable hash of the
scope. This isolates version `2.0` embeddings, clusters, and judged donors from
legacy representations and differently configured deployments.

Changing allocation weights, floors, ordering, or schema requires a new
representation version.

## Stacked Rollout

This branch is based on `stangoodwin/fullsessionembeddings`, and its pull request
must use that branch as its base. After the full-session embedding pull request
merges:

```powershell
gh pr edit <compression-pr> --base main
git fetch origin
git rebase origin/main
git push --force-with-lease origin stangoodwin/judge-session-compression
```

Do not use plain `--force`.

The Cosmos partition key now includes semantic scope. Existing documents using
the earlier `{vector_space_id}|{agent_id}` key are intentionally excluded and
require either a clean vector-space cutover or an explicit backfill before
deployment. Stale-document sweeps run across all semantic scopes so old version
data does not accumulate after cutover.

## Limitations

- UTF-8 bytes are not exact model-token counts.
- Prefix extraction can omit decisive content late in one field.
- A 32-byte extract establishes presence, not semantic sufficiency.
- Structurally extreme traces have no automatic fallback.
- The implementation does not claim equivalence to full-context human or judge
  evaluation.