# Security Audit Workflow for Annie

This repository uses automated dependency scanning and SBOM generation to mitigate supply chain risks.

## Automated Dependency Scanning
- Uses [`cargo audit`](https://github.com/RustSec/cargo-audit) in CI to detect vulnerable dependencies.
- Run manually with:
  ```bash
  cargo audit
  ```

## SBOM Generation
- Uses [`cargo sbom`](https://github.com/anchore/syft) to generate a Software Bill of Materials.
- Run manually with:
  ```bash
  cargo sbom -o sbom.spdx.json
  ```

## Dependency Version Pinning
- All critical dependencies are pinned to specific versions in `Cargo.toml`.
- Avoid using wildcards or loose version ranges.

## License Compliance
- Use [`cargo-license`](https://github.com/onur/cargo-license) to review dependency licenses.

## Regular Updates
- Review and update dependencies regularly.
- Consider enabling Dependabot for automated update PRs.

---
For more details, see `.github/workflows/security-audit.yml`.
