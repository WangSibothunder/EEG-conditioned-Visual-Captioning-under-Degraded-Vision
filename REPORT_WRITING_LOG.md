# Report Writing Log

## 2026-06-18

### Phase 1 (commit: 74c9152) — abstract + intro + related_work
- Compiled: 43 pages, xelatex clean
- Abstract: 470 chars (target 500-700)
- Ch1: 1484 chars (target 2200-2600) — restructured to 4 subsections
- Ch2: 1874 chars (target 2800-3300) — deduplicated, removed internal jargon
- Key changes: student tone, no Day3/Route5 references, moved research status to Ch2
### Phase 2 (commit: a3fa0b4) — dataset + system design
- Ch3: 1166 chars; Ch4: 983 chars. Three-line tables, removed internal jargon.

### Phases 7-10 (commit: b43bd4a) — Polish + 3 variant versions + bugfix
- **Critical fix**: `\end{tabular}` → `\end{tabularx}` in Ch8 — unclosed environment caused PDF truncation at page 18
- Fixed PDFs now 34-35 pages, zero LaTeX errors
- Created 4 complete report versions:
  - report_baseline: 35p, student course design tone
  - report_v1_practical: 34p, first-person, engineering focus
  - report_v2_academic: 34p, formal academic tone
  - report_v3_balanced: 35p, balanced practical/academic
- Total Chinese chars: ~12,057 (baseline body)
- All source code + PDFs uploaded to release v2026-06-18-final
- PDF preview: https://github.com/WangSibothunder/EEG-conditioned-Visual-Captioning-under-Degraded-Vision/releases/tag/v2026-06-18-final
