# GitHub Pages Workflow Note

The repository previously used a GitHub Pages deployment workflow with `actions/configure-pages`, `actions/upload-pages-artifact`, and `actions/deploy-pages`.

That deployment can fail with `Get Pages site failed` when GitHub Pages is not enabled for the repository or not configured to build using GitHub Actions.

The workflow has therefore been changed to a documentation check until Pages is explicitly enabled in the repository settings.

## Current behavior

The workflow now checks that the main research documentation files exist:

- `README.md`
- `docs/phd_proposal.md`
- `docs/research_roadmap.md`
- `docs/experiment_matrix.md`
- `configs/phd_research_program.yaml`
- `templates/experiment_report_template.md`
- `templates/failure_case_analysis.md`

## To enable Pages later

1. Go to repository settings.
2. Enable GitHub Pages.
3. Select GitHub Actions as the build and deployment source.
4. Restore the Pages deployment steps.

The insecure Node 20 fallback should not be used for this project.
