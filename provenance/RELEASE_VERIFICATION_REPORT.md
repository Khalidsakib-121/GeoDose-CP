# GeoDose-CP GitHub-ready publication package verification

- Package purpose: consolidated publication-facing code + compact data provenance.
- Python source files: **232**.
- Python AST/syntax failures: **0**.
- Obvious credential/private-key signature hits: **0**.
- Raw geospatial/image/data files in this compact package: **0**.
- Files covered by `PUBLICATION_FILE_SHA256.csv`: **473**.
- MineDoseBench v1.2 source-inventory entries included byte-for-byte in the publication core: **15 / 33**.
- The accepted MineDoseBench v1.2 benchmark release identity is recorded separately in `MINEDOSEBENCH_V1_2_ACCEPTED_RELEASE_IDENTITY.json`; Stage5B uses that frozen release.
- No obsolete unchanged v1.1 MineDoseBench freeze package is presented as the final benchmark implementation.
- No `SOURCE_GAP.md` is included.
- Large third-party NSW/DEA/SILO/TERN/GA source bytes are intentionally excluded; their authoritative source/provenance records are included under `data/` and `provenance/`.

## Important interpretation

This is a **publication-facing reproducibility package**, not a claim that every historical development wrapper from every intermediate archive is reproduced byte-for-byte. For MineDoseBench, it preserves the hash-verified scientific core available from the accepted v1.2 inventory and the exact accepted v1.2 release authorities used by Stage5B.
