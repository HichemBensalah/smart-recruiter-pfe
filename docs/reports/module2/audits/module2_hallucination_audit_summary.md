# Module 2 Hallucination Audit Summary

## Global Results
- Success profiles analyzed: 86
- Profiles with critical hallucinations: 14
- Profiles with major/medium hallucinations: 38
- Profiles clean or acceptable: 34
- Reliable profiles (keep): 34 (39.53%)
- Exclude from matching: 27
- Review before matching: 25

## Top Error Types
- summary_unsupported: 58
- experience_responsibility_unsupported: 49
- experience_unsupported: 40
- education_unsupported: 25
- identity_template_value: 15
- generic_template_field: 15
- identity_unsupported: 4

## Matching Readiness
- Verdict: Module 2 is not reliable enough to feed matching blindly; filtering is required first.
- Profiles that should be excluded from matching now: 27
- MongoDB risk: existing stored Module 2 profiles likely include risky records if they were imported without an audit gate.
- FAISS recommendation: do not index all Module 2 profiles blindly; exclude risky profiles first.
- Quality gate recommendation: yes, add a gate before matching and before FAISS indexing.

## Concrete Examples
- data\profile_builder_official_module2_rerun_ollama_fixed\pdf\CV32.json | bio.full_name | critical | full_name not clearly supported by Module 1 source. | value=None
  Evidence snippet: - Office:
- data\profile_builder_official_module2_rerun_ollama_fixed\pdf\CV32.json | expertise.summary | minor | Summary adds facts not clearly present in Module 1 source. | value=Data Visualization, Scientific Computing, Renewable Energy Systems
  Evidence snippet: - Departmentof Computer Science.
- data\profile_builder_official_module2_rerun_ollama_fixed\pdf\CV32.json | experiences[0].end_date | major | Experience end_date not clearly supported by source. | value=null
  Evidence snippet: - Cell:
- data\profile_builder_official_module2_rerun_ollama_fixed\pdf\CV34.json | bio.email | critical | Placeholder or template-like value used as real email. | value=info@qwikresume.com
- data\profile_builder_official_module2_rerun_ollama_fixed\pdf\CV34.json | expertise.summary | minor | Summary adds facts not clearly present in Module 1 source. | value=Data Architect in a collaborative environment utilizing creativity and technical skills for data modeling, analytics, and solution architecture expertise.
  Evidence snippet: Toobtaln apositlonasa Data Architectina collaborativeenvironment utilizing my creativity and technlcal skills.
- data\profile_builder_official_module2_rerun_ollama_fixed\pdf\CV34.json | experiences[0].start_date | major | Experience start_date not clearly supported by source. | value=2012-04
  Evidence snippet: ABCCorporation-2012-2014
- data\profile_builder_official_module2_rerun_ollama_fixed\docx\1_anonyme.json | bio.full_name | critical | Placeholder or template-like value used as real full_name. | value=null
- data\profile_builder_official_module2_rerun_ollama_fixed\docx\1_anonyme.json | bio.location | major | Placeholder or template-like value used as real location. | value=null
- data\profile_builder_official_module2_rerun_ollama_fixed\docx\1_anonyme.json | bio.full_name | critical | Template/example value leaked into final profile. | value=null
- data\profile_builder_official_module2_rerun_ollama_fixed\images\2_image.json | bio.email | critical | Placeholder or template-like value used as real email. | value=email@youremail.com
- data\profile_builder_official_module2_rerun_ollama_fixed\images\2_image.json | expertise.summary | major | Summary adds facts not clearly present in Module 1 source. | value=Data analysis and modeling in streaming platforms using statistical techniques.
  Evidence snippet: Springboard Data Science Course-Data Science Capstone Project:Specialized Analysisof Streaming Platform Patterns
- data\profile_builder_official_module2_rerun_ollama_fixed\images\2_image.json | experiences[0].city | major | Experience city not clearly supported by source. | value=Unknown
  Evidence snippet: SKILLS & KNOWLEDGE

## Notes
- This audit is read-only against existing Module 2 success outputs and their linked Module 1 artifacts.
- The main failure mode observed is abstractive reconstruction on top of noisy OCR, not just minor formatting drift.
- Weak evidence means the field may be partially supported but not confidently grounded as written in the JSON.