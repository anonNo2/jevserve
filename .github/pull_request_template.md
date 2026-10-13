<!-- 可用中文填写。Keep this proportional to the change; mark irrelevant gates N/A
with a reason. Never paste credentials, private prompts, weights or private logs. -->

### Problem and resulting behavior

Describe the concrete trigger and before/after behavior. Link related issues.

### Validation and evidence

List exact commands, outcomes and reviewed artifact/report links. Distinguish CPU,
image build, real GPU execution, probability parity and Docker/Compose checks.
State failures and unrun checks explicitly; use N/A with a reason when appropriate.

### Risks and remaining work

Describe compatibility, model semantics, dependency/license or operational changes.
For model changes, include the checklist from docs/contributing-models.md, pinned
source/model/image identities, frozen tolerances, precision and reference scope.

- [ ] Behavior/contract tests cover meaningful changed failure paths where needed.
- [ ] Model order, masks, temperature and probability semantics are preserved, or
      an intentional change is explained with reference evidence.
- [ ] Documentation and English/Chinese README capability claims agree where affected.
- [ ] Task evidence and next steps are recorded in docs/development/STATUS.md;
      material decisions are recorded in DECISIONS.md where applicable.
- [ ] No weights, credentials, private data, caches or host identifiers are included;
      reused code/dependencies retain license and attribution.
- [ ] No unrun gate is marked passed and no new certification/release is implied.
