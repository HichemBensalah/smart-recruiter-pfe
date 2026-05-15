# Patch Dry-Run Report V2 Safe URL Fixes

- `total_profiles_checked`: 90
- `profiles_that_would_change`: 0
- `urls_that_would_be_fixed`: 0
- `skipped_due_to_quality_regression`: 0
- `reliability_regressions_count`: 0
- `safe_to_execute`: true

## Risk Before

```json
{
  "low": 89,
  "medium": 1
}
```

## Risk After Estimated

```json
{
  "low": 89,
  "medium": 1
}
```

## Examples Of URL Fixes

```json
[]
```

## Recommendation

Dry-run looks safe. Apply --execute only after reviewing the examples_of_url_fixes and confirming that only URL normalizations should be persisted.
