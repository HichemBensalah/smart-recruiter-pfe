# CrossEncoder Comparison Report

- `generated_at_utc`: 2026-05-07T13:19:47.244611+00:00
- `baseline_compared_to`: baseline1_faiss_matching_v3
- `hichem_remains_top1`: true
- `top1_score_before`: 0.8172
- `top1_score_after`: 0.7078

## Top 10 Baseline FAISS

- #1 Hichem Bensalah | final=0.8172 | faiss_rank=None | ce_rank=None | risk=low
- #2 MOHAMED AZIZ BELAWEID | final=0.5528 | faiss_rank=None | ce_rank=None | risk=low
- #3 Candidate (ID: candidate_8eea1b635447) | final=0.5466 | faiss_rank=None | ce_rank=None | risk=medium
- #4 JEFFERYGORCZANY | final=0.4332 | faiss_rank=None | ce_rank=None | risk=low
- #5 Candidate (ID: candidate_71e03ea99985) | final=0.3502 | faiss_rank=None | ce_rank=None | risk=low
- #6 Candidate (ID: candidate_1d475044c93c) | final=0.3464 | faiss_rank=None | ce_rank=None | risk=low
- #7 Markus Rohan | final=0.3181 | faiss_rank=None | ce_rank=None | risk=medium
- #8 Candidate (ID: candidate_c564b8eceb3d) | final=0.2731 | faiss_rank=None | ce_rank=None | risk=medium
- #9 MILDREDZEMLAK | final=0.2628 | faiss_rank=None | ce_rank=None | risk=low
- #10 Justine Hendrickson | final=0.2573 | faiss_rank=None | ce_rank=None | risk=low

## Top 10 FAISS + CrossEncoder

- #1 Hichem Bensalah | final=0.7078 | faiss_rank=2 | ce_rank=12 | risk=low
- #2 Candidate (ID: candidate_8eea1b635447) | final=0.6184 | faiss_rank=10 | ce_rank=16 | risk=low
- #3 MOHAMED AZIZ BELAWEID | final=0.5479 | faiss_rank=20 | ce_rank=10 | risk=low
- #4 JEFFERYGORCZANY | final=0.46 | faiss_rank=3 | ce_rank=2 | risk=low
- #5 Markus Rohan | final=0.4085 | faiss_rank=11 | ce_rank=11 | risk=low
- #6 Candidate (ID: candidate_c564b8eceb3d) | final=0.3652 | faiss_rank=18 | ce_rank=5 | risk=low
- #7 Candidate (ID: candidate_71e03ea99985) | final=0.3393 | faiss_rank=12 | ce_rank=13 | risk=low
- #8 MILDREDZEMLAK | final=0.3229 | faiss_rank=6 | ce_rank=1 | risk=low
- #9 Karina Blick | final=0.2792 | faiss_rank=14 | ce_rank=3 | risk=low
- #10 Justine Hendrickson | final=0.2729 | faiss_rank=9 | ce_rank=4 | risk=low

## Candidates Moved Up

```json
[
  {
    "candidate_id": "candidate_8eea1b635447",
    "full_name": "Candidate (ID: candidate_8eea1b635447)",
    "before_rank": 3,
    "after_rank": 2,
    "delta": 1
  },
  {
    "candidate_id": "candidate_073b7a3d39ba",
    "full_name": "Markus Rohan",
    "before_rank": 7,
    "after_rank": 5,
    "delta": 2
  },
  {
    "candidate_id": "candidate_c564b8eceb3d",
    "full_name": "Candidate (ID: candidate_c564b8eceb3d)",
    "before_rank": 8,
    "after_rank": 6,
    "delta": 2
  },
  {
    "candidate_id": "candidate_d813a4aedd03",
    "full_name": "MILDREDZEMLAK",
    "before_rank": 9,
    "after_rank": 8,
    "delta": 1
  }
]
```

## Candidates Moved Down

```json
[
  {
    "candidate_id": "candidate_206d746034ef",
    "full_name": "MOHAMED AZIZ BELAWEID",
    "before_rank": 2,
    "after_rank": 3,
    "delta": -1
  },
  {
    "candidate_id": "candidate_71e03ea99985",
    "full_name": "Candidate (ID: candidate_71e03ea99985)",
    "before_rank": 5,
    "after_rank": 7,
    "delta": -2
  }
]
```

## New / Exited Top 10

```json
{
  "new_candidates_in_top_10": [
    {
      "rank": 9,
      "candidate_id": "candidate_664415ab2fe1",
      "full_name": "Karina Blick",
      "final_score": 0.2792,
      "faiss_rank": 14,
      "faiss_score": 0.4504,
      "cross_encoder_rank": 3,
      "cross_encoder_score": -2.65295,
      "profile_kind": "complete_profile",
      "hallucination_risk": "low",
      "must_have_coverage": 0.2,
      "matched_skills": [
        "Python"
      ],
      "missing_required_skills": [
        "FastAPI",
        "MongoDB",
        "Docker",
        "REST API"
      ]
    }
  ],
  "candidates_exited_top_10": [
    {
      "rank": 6,
      "candidate_id": "candidate_1d475044c93c",
      "full_name": "Candidate (ID: candidate_1d475044c93c)",
      "final_score": 0.3464,
      "faiss_rank": null,
      "faiss_score": null,
      "cross_encoder_rank": null,
      "cross_encoder_score": null,
      "profile_kind": "complete_profile",
      "hallucination_risk": "low",
      "must_have_coverage": 0.4,
      "matched_skills": [
        "Python",
        "MongoDB"
      ],
      "missing_required_skills": [
        "FastAPI",
        "Docker",
        "REST API"
      ]
    }
  ]
}
```

## Risk And Skill Coverage

```json
{
  "medium_risk_ranks_before": {
    "candidate_8eea1b635447": 3,
    "candidate_073b7a3d39ba": 7,
    "candidate_c564b8eceb3d": 8
  },
  "medium_risk_ranks_after": {},
  "high_skill_coverage_candidates_after": [
    {
      "rank": 1,
      "candidate_id": "candidate_1487f3187f7b",
      "full_name": "Hichem Bensalah",
      "final_score": 0.7078,
      "faiss_rank": 2,
      "faiss_score": 0.593,
      "cross_encoder_rank": 12,
      "cross_encoder_score": -3.683536,
      "profile_kind": "complete_profile",
      "hallucination_risk": "low",
      "must_have_coverage": 0.8,
      "matched_skills": [
        "Python",
        "FastAPI",
        "MongoDB",
        "Docker"
      ],
      "missing_required_skills": [
        "REST API"
      ]
    },
    {
      "rank": 2,
      "candidate_id": "candidate_8eea1b635447",
      "full_name": "Candidate (ID: candidate_8eea1b635447)",
      "final_score": 0.6184,
      "faiss_rank": 10,
      "faiss_score": 0.4646,
      "cross_encoder_rank": 16,
      "cross_encoder_score": -4.173741,
      "profile_kind": "complete_profile",
      "hallucination_risk": "low",
      "must_have_coverage": 0.8,
      "matched_skills": [
        "Python",
        "FastAPI",
        "MongoDB",
        "Docker"
      ],
      "missing_required_skills": [
        "REST API"
      ]
    }
  ]
}
```

## Qualitative Analysis

CrossEncoder preserves the strongest top candidate while giving the final scorer a finer semantic signal. The new top 10 contains 2 candidates with >=0.8 must-have coverage and 0 medium-risk candidates.
