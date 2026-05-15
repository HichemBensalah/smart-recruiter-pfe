# Patch Dry-Run Report V2 Fixes

- `total_profiles_checked`: 90
- `profiles_that_would_change`: 3
- `emails_that_would_be_fixed`: 0
- `urls_that_would_be_fixed`: 4
- `years_experience_that_would_be_recovered`: 0
- `experience_level_that_would_be_recovered`: 0
- `high_risk_remaining`: 0
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
  "low": 88,
  "medium": 2
}
```

## Examples Of Changes

```json
[
  {
    "profile_file": "data\\profile_builder_module2_v2_grounded_all\\profiles\\grounded_profiles\\docx_Aziz_resumer.json",
    "artifact_path": "data\\processed_official_module1\\docx\\Aziz_resumer.json",
    "changes": [
      {
        "field": "profile.bio.linkedin",
        "before": "linkedin.com/in/mohamed-aziz-belaweid/⋄",
        "after": "https://www.linkedin.com/in/mohamed-aziz-belaweid/"
      }
    ]
  },
  {
    "profile_file": "data\\profile_builder_module2_v2_grounded_all\\profiles\\grounded_profiles\\pdf_AnuvaGoyal_Latex.json",
    "artifact_path": "data\\processed_official_module1\\pdf\\AnuvaGoyal_Latex.json",
    "changes": [
      {
        "field": "grounding.hallucination_risk",
        "before": "low",
        "after": "medium"
      },
      {
        "field": "grounding.reliability_score",
        "before": 0.9436,
        "after": 0.792
      }
    ]
  },
  {
    "profile_file": "data\\profile_builder_module2_v2_grounded_all\\profiles\\grounded_profiles\\pdf_Aziz_resume.json",
    "artifact_path": "data\\processed_official_module1\\pdf\\Aziz_resume.json",
    "changes": [
      {
        "field": "profile.bio.linkedin",
        "before": "linkedin.com/in/mohamed-aziz-belaweid/⋄",
        "after": "https://www.linkedin.com/in/mohamed-aziz-belaweid/"
      }
    ]
  }
]
```

## Recommendation

Dry-run looks safe. Apply --execute only after reviewing the examples_of_changes and confirming that the contact and experience consolidations are acceptable.
