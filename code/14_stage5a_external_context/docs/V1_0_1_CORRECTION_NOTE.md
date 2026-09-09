# v1.0.1 correction note

This release supersedes External Context Production v1.0.0 FREEZE CANDIDATE.

The v1.0.0 production gate correctly verified all 37 immutable base files individually, but then recomputed the base Acquisition v1.0.1 aggregate with the *production package's* aggregate rule (`file+sha256`). The accepted Acquisition v1.0.1 package uses a different, explicit aggregate contract: `file<TAB>size<TAB>sha256<LF>` for each immutable manifest row, including the final newline. Therefore v1.0.0 failed even on an intact accepted acquisition folder.

v1.0.1 fixes only this operational authority-verification defect and strengthens tests so the actual acquisition working folder is verified during the production runner before extraction begins. Scientific authorities, 23,618/92 substrate partition, 0.99 threshold, 18 governing layers, source identities, formulas, and outcome-blind boundary are unchanged.
