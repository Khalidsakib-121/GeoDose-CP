# MineDoseBench publication core

This directory is the publication-facing MineDoseBench code/authority bundle used with GeoDose-CP.

- The scientific Python modules, schema, evaluation contract, requirements, and method-protocol documents in this directory are byte-for-byte matches to entries in the accepted MineDoseBench v1.2 source inventory.
- `accepted_v1_2_release_metadata/` contains metadata and verification records extracted from the exact accepted `GeoDose_MineDoseBench_v1_2_0_RELEASE_CANDIDATE.zip` consumed by Stage5B.
- Stage5B evaluates the frozen accepted v1.2 benchmark release; it does not rerun the obsolete unchanged v1.1 benchmark freeze.
- The external-context production stage is separately preserved in `../14_stage5a_external_context/` and is the authority for the 23,618 context-complete substrate and 92 support exclusions.

This publication package intentionally avoids labeling reconstructed historical wrapper scripts as original source. The scientific core and the accepted release authorities are preserved directly instead.
