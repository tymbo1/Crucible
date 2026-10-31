# CNSS Partner Generalization v1 - CPU result

Classification: INCONCLUSIVE_PREREQUISITE_FAILURE

Skills: A, B, C, D; seeds: 42, 1337, 2025; lambda values: 0.1, 0.4, 0.9, 1.0.

Development partnerships: A->B, B->C, C->D, D->A, A->C, B->D.
Held-out test partnerships: B->A, C->B, D->C, A->D, C->A, D->B.

| Policy | Held-out collision-free mean | Weakest held-out partnership mean |
| --- | ---: | ---: |
| global | 0.6719 | 0.2941 |
| sender_only | 0.6691 | 0.2941 |
| receiver_only | 0.6475 | 0.2941 |
| skill_shared | 0.6734 | 0.2941 |

Trained skill adapters: 48. Pair/lambda/seed evaluations: 576.

The preregistered individual-competence gate was not passed. Skill D (modular addition by 0x37) reached at most 0.6875 exact test accuracy across the four lambda values and three seeds, below the required 0.95. Therefore the held-out partnership results are descriptive only and cannot adjudicate sender-specific, receiver-specific, or pair-specific compatibility.

The complete sender-lambda/receiver-lambda tensor is descriptive. No held-out partnership was used to select a deployable policy.

This result is limited to the registered 8-bit mechanism probe and does not establish broad reasoning, lifelong learning, or scalability.

Machine-readable details: CNSS_PARTNER_GENERALIZATION_RESULT.json.
