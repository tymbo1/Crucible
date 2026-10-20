# Crucible sustained-equivalence gate — RunPod hold

Date: 2026-10-20
Status: HOLD — waiting for secure RTX 4090 capacity

## Frozen execution package

Archive:

`/home/timbushnell/runpod_sustained_equivalence_gate_20261020_v1.tar`

Size: 1,910,097,920 bytes

SHA-256:

`e235b583f6f256b40ef165eb41e13fb23a9b3ca7ee0d4c90780a6d82f4abdd57`

The package contains the frozen 128-byte data and anchors, the unchanged context runner, the original backend, the rank-preserving grouped backend, and optimizer-state hashing support.

## Deployment attempts

Two secure RTX 4090 `podFindAndDeployOnDemand` requests were made for the authorized gate. Both returned RunPod `SUPPLY_CONSTRAINT`:

> There are no longer any instances available with the requested specifications. Please refresh and try again.

Neither response returned a pod ID. No community pod or RTX 5090 fallback was attempted.

## Billing/resource check

A read-only account query after the failed attempts showed no new running or billable pod. All listed pods were `EXITED`; the new gate name was absent. No compute credit was consumed by these failed capacity requests.

## Hold instruction

Do not deploy again without explicit authorization. Do not switch to a community instance or RTX 5090 for this gate. When authorized and secure RTX 4090 capacity is available, upload this exact archive, verify its hash remotely, run the staged 10-step and 100-step checks before the 3,125-step paired validation, collect artifacts, and terminate the pod only after collection and verification.

## Read-only availability recheck

The subsequent `runpodctl gpu list` query reports:

- RTX 4090: available
- Secure cloud: available
- Secure price: US$0.89/hour
- Overall stock: Low
- Listed secure datacenter: `EU-CZ-1`

No deployment request was made during this recheck. The gate remains on hold pending explicit launch authorization.
