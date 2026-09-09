# Windows path-resolution correction — v1.0.1

The v1.0 runner expected the M3/M4 accepted output ZIPs directly under the outer extraction directory. The distributed M3/M4 source ZIPs each contain a same-named top-level project folder. When extracted by Windows Explorer into a same-named destination folder, the normal on-disk layout is therefore double-nested.

This release checks the double-nested layout first and the single-nested layout second. It also lists all attempted paths if a frozen input cannot be found.

This correction changes path discovery only. It does not modify M1-M6 definitions, accepted M3/M4 results, D1/D2 mathematics, Stage3C baseline bytes, reductions, thresholds, or experiments.
