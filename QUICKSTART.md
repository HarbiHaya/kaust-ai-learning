# Quick start

## Open the exam

After GitHub Pages is enabled, open:

`https://harbihaya.github.io/kaust-ai-learning/`

The Day 1 exam saves answers in the current browser. Returning with the same browser restores the attempt.

## Capture a thought immediately

Open [`notes/inbox.md`](notes/inbox.md) and add a line. Keep capture faster than organization.

Examples:

```text
- Why can ROC-AUC look good on an imbalanced dataset?
- Revisit leakage caused by preprocessing before cross-validation.
- My explanation of macro F1 is still unclear.
```

## Write structured notes

Copy [`templates/day-notes.md`](templates/day-notes.md) into the relevant day folder. Day 1 already exists at [`notes/stage-2/day-01/README.md`](notes/stage-2/day-01/README.md).

## Add another interactive exam

1. Add the question bank under `exams/day-XX/` using [`templates/exam-spec.md`](templates/exam-spec.md).
2. Ask Codex: “Convert `exams/day-XX/questions.md` into an interactive exam matching the Day 1 design.”
3. Put the resulting page under `docs/exams/day-XX/` or add it to the main learning page.
4. Commit and push to `main`.

## Publish changes

```bash
git add .
git commit -m "Add Day 2 notes and exam"
git push
```

GitHub Pages republishes changes from the `docs/` directory.

