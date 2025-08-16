```markdown
# Security Audit Workflow for Annie

This repository uses automated dependency scanning, SBOM generation, and enhanced security tests to mitigate supply chain risks and ensure robust input validation.

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

## Enhanced Security Tests
- **Input Validation**: New fuzz tests ensure that input validation is robust, checking for conditions such as invalid dimensions and excessive allocations.
- **Concurrency Stress Testing**: Tests added to ensure thread safety and proper handling of concurrent operations.
- **GPU Error Conditions**: Tests to verify error handling for GPU operations, including mismatched dimensions and large allocations.
- **Boundary and DoS Checks**: Security tests to ensure boundary conditions are respected and to prevent denial-of-service attempts through excessive resource allocation.

---
For more details, see `.github/workflows/security-audit.yml`.
```