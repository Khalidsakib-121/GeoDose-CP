# Scientific contract

1. The support-census decision is now the prospective authority for context availability.
2. Retain exactly 23,618 context-complete blocks and exclude exactly 92 blocks that failed one or more of the 18 frozen >=0.99 source-support gates.
3. No threshold relaxation, interpolation, nearest-neighbor filling, mine-level filling, synthetic context, alternative-source substitution, or post-outcome recovery is allowed.
4. Production uses the exact cached source/derived raster bytes already recorded by Support Census v1.0.1. This prevents source drift between support selection and value extraction.
5. Every final value is an area-weighted frozen-polygon extraction in EPSG:9473 under the already frozen climate/soil/terrain transformations.
6. This stage reads no MineDoseBench outcomes and no M2-M6 performance.
7. This stage does not run MineDoseBench FREEZE. It produces a verified handoff for a prospectively revised Stage5A freeze package.
