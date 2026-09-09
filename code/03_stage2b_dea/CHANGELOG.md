# Changelog

## 1.2.0-local-stac

- Preserved `graph_component_id` as its frozen string identifier instead of incorrectly casting it to an integer.
- Updated the self-test to use a realistic component ID such as `M1_CC001`.
- Added independent verification that component IDs are present and retain the `_CC` identifier structure.
- No changes to DEA products, years, masks, grid, graph, folds, exposure, quality gates, or mine-selection rules.

## 1.1.0-local-stac

- Forced DEA STAC item searches to `method="GET"`.
- Replaced complex `intersects` POST bodies with documented WGS84 `bbox` queries.
- Added five bounded search retries (5, 15, 30, 60 seconds).
- Added explicit STAC client timeouts and informative final errors.
- Recorded and independently verified the GET/bbox search policy.
- Added Python 3.10 virtual-environment batch launchers.

## 1.0.0-local-stac

- Initial public DEA STAC/AWS implementation.
