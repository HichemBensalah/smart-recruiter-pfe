# Matching Ranking Comparison Grounded V3

## Top 10 V1

- #1 Hichem Bensalah | score=0.5749 | profile_kind=complete_profile | risk=None | display_name_quality=None
- #2 MOHAMED AZIZ BELAWEID | score=0.3537 | profile_kind=complete_profile | risk=None | display_name_quality=None
- #3 JefferyGorczany | score=0.2342 | profile_kind=partial_profile | risk=None | display_name_quality=None
- #4 None | score=0.229 | profile_kind=partial_profile | risk=None | display_name_quality=None
- #5 Wenzhe(Evelyn)Xu | score=0.228 | profile_kind=partial_profile | risk=None | display_name_quality=None
- #6 JESSICACLAIRE | score=0.2267 | profile_kind=complete_profile | risk=None | display_name_quality=None
- #7 None | score=0.2262 | profile_kind=complete_profile | risk=None | display_name_quality=None
- #8 MILDREDZEMLAK | score=0.2259 | profile_kind=complete_profile | risk=None | display_name_quality=None
- #9 None | score=0.2258 | profile_kind=partial_profile | risk=None | display_name_quality=None
- #10 None | score=0.2253 | profile_kind=partial_profile | risk=None | display_name_quality=None

## Top 10 Grounded V2

- #1 Hichem Bensalah | score=0.7671 | profile_kind=complete_profile | risk=None | display_name_quality=None
- #2                 | score=0.763 | profile_kind=complete_profile | risk=None | display_name_quality=None
- #3 MOHAMED AZIZ BELAWEID | score=0.5362 | profile_kind=complete_profile | risk=None | display_name_quality=None
- #4 JEFFERYGORCZANY | score=0.4139 | profile_kind=complete_profile | risk=None | display_name_quality=None
- #5 Data Scientist | score=0.3881 | profile_kind=complete_profile | risk=None | display_name_quality=None
- #6 RESUME OBJECTIVE | score=0.3867 | profile_kind=complete_profile | risk=None | display_name_quality=None
- #7 from Resume Genius | score=0.3764 | profile_kind=complete_profile | risk=None | display_name_quality=None
- #8 Markus Rohan | score=0.3712 | profile_kind=partial_profile | risk=None | display_name_quality=None
- #9 Justine Hendrickson | score=0.2355 | profile_kind=complete_profile | risk=None | display_name_quality=None
- #10 MILDREDZEMLAK | score=0.2316 | profile_kind=complete_profile | risk=None | display_name_quality=None

## Top 10 Grounded V3

- #1 Hichem Bensalah | score=0.8172 | profile_kind=complete_profile | risk=low | display_name_quality=ok
- #2 MOHAMED AZIZ BELAWEID | score=0.5528 | profile_kind=complete_profile | risk=low | display_name_quality=ok
- #3 Candidate (ID: candidate_8eea1b635447) | score=0.5466 | profile_kind=complete_profile | risk=medium | display_name_quality=weak
- #4 JEFFERYGORCZANY | score=0.4332 | profile_kind=complete_profile | risk=low | display_name_quality=ok
- #5 Candidate (ID: candidate_71e03ea99985) | score=0.3502 | profile_kind=complete_profile | risk=low | display_name_quality=weak
- #6 Candidate (ID: candidate_1d475044c93c) | score=0.3464 | profile_kind=complete_profile | risk=low | display_name_quality=weak
- #7 Markus Rohan | score=0.3181 | profile_kind=partial_profile | risk=medium | display_name_quality=ok
- #8 Candidate (ID: candidate_c564b8eceb3d) | score=0.2731 | profile_kind=complete_profile | risk=medium | display_name_quality=weak
- #9 MILDREDZEMLAK | score=0.2628 | profile_kind=complete_profile | risk=low | display_name_quality=ok
- #10 Justine Hendrickson | score=0.2573 | profile_kind=complete_profile | risk=low | display_name_quality=ok

## Key Findings

- Suspect display names in V2: 4
- Suspect display names in V3: 4
- New candidates in V3 top 10: 8
- Removed from top 10 in V3: 8
- Skill normalizer effect: Canonical aliases reduce fragmentation such as Fast API -> FastAPI and REST API design -> REST API.
- Grounded quality effect: V3 applies reliability, profile_kind, hallucination_risk and nullified-field penalties on top of FAISS similarity.
- Display-name fix: Suspicious names are replaced with Candidate (ID: ...) instead of surfacing OCR titles or template text.
- High-risk penalty effect: Medium and high hallucination-risk profiles receive softer score multipliers rather than exclusion.