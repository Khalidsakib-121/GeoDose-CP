SUPERSEDES v1.0.0: v1.0.1 corrects the base-manifest aggregate recomputation contract; scientific design is unchanged.

# Release recheck

Before packaging v1.0.1 FREEZE CANDIDATE:

- Embedded accepted Support Census ZIP SHA-256 matched `2facc83842bd2ec8e30a2763c7040cb5a6e7446a0292d52d1eff0a1ea9c5152e`.
- Census output manifest aggregate matched `053e09d0ab54fad015c5c2787029ff8bbb1b28195b31115f7ae76cfe6cd68177` and every internal file hash/size was rechecked.
- Exact prospective partition rechecked: 23,710 original = 23,618 retained + 92 excluded; disjoint and exhaustive.
- Frozen support rule remains >=0.99 over exactly 18 governing layers.
- Exclusion pattern rechecked: Liddell 89, Bulga 3, other mines 0.
- Source manifest rechecked: 4 SILO + 13 TERN + GA raw/derived provenance; credentials not serialized.
- Production source contains no live network client. It requires exact census-locked AOI/derived source bytes and fails closed on missing/hash-mismatched cache.
- Production unit/regression suite: 26/26 PASS.
- Developer synthetic replay passed for exact polygon-area subset invariance and circular terrain-aspect aggregation.
- Python source compilation passed.
- Downstream compatibility rechecked: unchanged MineDoseBench Generator v1.1.0 requires all 23,710 original rows, therefore this production package stops pending a prospectively revised Stage5A freeze package rather than falsely authorizing the old generator.
