# Acceptance Checklist

A report is acceptable only when all checks pass.

## Scientific checks

- Exact aggregate input path is stated.
- Method labels match generator outputs.
- Claims are bounded to the evaluated cohort and budget.
- Judged labels and imputed values are explicitly separated.
- Replay/bootstrap envelopes are not described as independent-label confidence intervals.
- Proxy coverage is labeled as cohort-relative and non-universal.

## Narrative checks

- Evidence follows the report template: decision/source, methods, data/design, visual results, analysis/limits, and validation.
- Illustrative examples are marked as illustrative.
- No fabricated winner percentages or unsupported concrete claims.
- Every tested method has a plain-language mechanism and assumptions, not just a name.
- Data origin, label source, exclusions, budgets, absolute caps, and repeat design are explicit.
- Charts have units, denominators, comparable axes, consistent method colors, and takeaways.
- Dataset/agent differences and accuracy/coverage/cost tradeoffs are not hidden by pooled results.

## Validation checks

- Local Playwright gate passes desktop and mobile for interactive HTML.
- Horizontal overflow check uses document-level tolerance and allows intentional scroll wrappers.
- Console errors, page errors, failed requests, blocked external requests, and broken images are all zero.
- If PDF requested: explicit --print-html source is used, output extension is .pdf, and overwrite protection is respected.
- Source HTML hashes are unchanged by validation.
- Fixed-A4 print content fits the printable width; manual PDF inspection covers page breaks, chart labels and numeric parity.
- Exercise interactive filters/controls and verify their displayed scope; the generic checker does not test those interactions.
