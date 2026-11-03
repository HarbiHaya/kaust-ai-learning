# KAUST AI Learning

A personal learning repository for KAUST Academy AI preparation. It collects original study notes, practice notebooks, question banks, interactive exams, and progress tools for Stage 1 and Stage 2.

## Current tools

- **Stage 2 Day 1 interactive exam:** 80 difficult questions with five choices, automatic scoring, section-level results, mistake review, and browser-persistent progress.
- **Pandas learning-by-doing notebooks:** practice and solution notebooks for data-analysis revision.
- **Study planning resources:** local planning and tracking tools used during exam preparation.

## Lowest-friction workflow

1. Put unfinished thoughts in [`notes/inbox.md`](notes/inbox.md). No formatting is required.
2. Move useful material into the matching Stage and Day folder when you have time.
3. Use [`templates/day-notes.md`](templates/day-notes.md) when you want structured notes.
4. Ask Codex to convert a question bank into another interactive exam matching Day 1.
5. Push to `main`. GitHub Pages republishes the `docs/` website automatically.

See [`QUICKSTART.md`](QUICKSTART.md) for copyable commands and the repository map.

## Website

The GitHub Pages source is in [`docs/`](docs/). After the repository is published, configure GitHub Pages to deploy from the `main` branch and `/docs` directory.

The exam stores answers in the visitor's browser. Refreshing or reopening the page on the same browser preserves progress.

## Suggested future structure

```text
docs/                         GitHub Pages website
notes/inbox.md                Fast, unstructured capture
notes/stage-1/                Original Stage 1 notes
notes/stage-2/                Original Stage 2 notes
exams/                        Question banks and mock exams
notebooks/                    Practice notebooks
projects/                     Small ML and data-science projects
resources/                    Links to official resources
templates/                    Reusable note and exam formats
```

## Repository rules

- Add original explanations, exercises, code, and links to official resources.
- Do not upload KAUST or Coursera slides, recordings, paid course files, private exam material, or copyrighted PDFs.
- Do not commit personal schedules, credentials, API keys, access tokens, or private student information.
- Keep answer keys separate from quiz interfaces whenever possible.

