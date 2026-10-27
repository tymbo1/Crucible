# Crucible 500M cross-domain reasoning curriculum plan

Date recorded: 2026-10-28  
Status: **FUTURE PLAN — NOT APPROVED FOR INGESTION OR TRAINING**

## Objective

Prepare a future curriculum and transfer-evaluation programme for a roughly
500-million-parameter Crucible model. The curriculum should test whether
verified learning transfers across mathematical, logical, scientific,
linguistic and normative domains rather than merely rewarding memorisation of
familiar answers.

This plan is separate from the frozen Track B B0–B3 procedure-proposer
experiment. It does not authorize dataset download, ingestion, model design,
training or evaluation.

## Candidate dataset mix

| Dataset | Primary domains | Intended role |
|---|---|---|
| MATH | Mathematics | Multi-step mathematics, algebra, geometry and proof-style problem solving |
| BIG-Bench Hard | Logic, language and reasoning | Diverse deduction, causal reasoning, ambiguity, tracking and symbolic tasks |
| SciBench | Science and mathematics | College-level physics, chemistry and mathematics problem solving |
| ETHICS | Ethics and normative reasoning | Justice, deontology, virtue ethics, utilitarian and commonsense morality tasks |
| BIG-Bench | Language, philosophy and conceptual reasoning | Selected interpretation, philosophical and conceptual reasoning tasks |

The exact dataset revisions, subsets, licenses, access terms and permitted
training uses must be verified before ingestion. No benchmark should be
treated as an authority on objective moral truth; ETHICS measures agreement
with its benchmark judgments under its task definitions.

## Proposed curriculum order

1. Formal reasoning and mathematics
2. Scientific reasoning
3. Language and interpretation
4. Ethics and normative reasoning
5. Philosophy and broader conceptual transfer

The ordering is a proposed experimental curriculum, not a claim that ethics,
philosophy or language are less important. Formal domains are placed earlier
because their verification criteria are generally more explicit.

## Cumulative adaptive-learning question

The future experiment should test whether acquiring verified procedures in
mathematics and formal logic improves later acquisition in scientific
reasoning, linguistic interpretation or ethical argument analysis without
providing answers in advance.

The curriculum must be converted into controlled learning episodes with:

- withheld task families and compositional splits;
- independent verification where the domain permits it;
- reset and post-reset execution tests;
- retention and interference checks;
- abstention and uncertainty measurement;
- provenance and licensing records;
- comparisons against frozen-learning and procedure-memory controls;
- separation of direct retrieval benefit from improvement in the induction
  strategy itself.

Performance on these datasets alone will not establish lifelong procedural
learning. The central outcome is whether verified experience improves the
learner's ability to acquire genuinely unfamiliar procedures later.

## Future 500M model boundary

The intended scale is approximately 500M parameters. Architecture, tokenizer,
optimizer, training budget, hardware and dataset mixture remain unspecified
and must be selected in a separate protocol after the curriculum data audit.

The Verified Learning Operator Library remains a separate future hypothesis;
this plan does not implement or authorize it.
