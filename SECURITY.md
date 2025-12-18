# Security Policy

## Overview

While `diffml` is primarily a research-oriented project focused on replicating differential machine learning experiments, we take security seriously. We welcome security reports and will do our best to address any vulnerabilities that could affect users of this library or its dependencies.

## Supported Versions

We provide security updates for the following versions:

| Version | Support Status |
| ------- | -------------- |
| `main` branch | ✅ **Actively supported** |
| Latest release (`v*.*.*`) | ✅ **Supported** |
| Older releases | ⚠️ **Best effort only** |

Security fixes will generally be applied to the `main` branch first, then backported to the latest release if applicable. Older releases receive security updates on a best-effort basis only.

## Reporting a Vulnerability

### Private Disclosure (Recommended for High/Critical Issues)

For sensitive security vulnerabilities, please use GitHub's **Security Advisories** feature:

1. Go to the repository's **Security** tab
2. Click on **Report a vulnerability**
3. Provide a detailed description of the vulnerability
4. Include steps to reproduce if applicable
5. Suggest a fix if you have one

This ensures the issue remains private while we work on a fix.

### Public Disclosure (Low Impact Issues)

For low-impact security issues that don't expose sensitive exploit details, you may:

1. Open a regular GitHub issue
2. Add the `security` label
3. Provide details without including exploit code

⚠️ **Important:** Never disclose sensitive exploit details, proof-of-concept attacks, or detailed reproduction steps for critical vulnerabilities in public issues.

## Response Expectations

- **Initial acknowledgement:** Within **1 week** of report submission
- **Status update:** Within **2 weeks** with our assessment
- **Fix timeline:** Depends on severity, typically within **30 days** for critical issues

Please note that as a research project maintained by volunteers, we cannot guarantee specific SLAs, but we commit to best-effort response and remediation.

## Security Best Practices for Users

When using this library:

1. **Keep dependencies updated:** Run `poetry update` regularly
2. **Use virtual environments:** Isolate project dependencies
3. **Review experiment outputs:** Be cautious when running experiments with untrusted data
4. **Monitor PyTorch security advisories:** Stay informed about PyTorch security updates

## Scope

Security issues we're interested in:

- Dependency vulnerabilities
- Code injection possibilities
- Unsafe deserialization
- Path traversal in file operations
- Memory safety issues in native extensions

Out of scope:

- Denial of service via resource exhaustion (expected in ML workloads)
- Numerical accuracy issues (unless exploitable)
- Performance issues

## Recognition

We appreciate security researchers who help improve this project. Contributors who report valid security issues will be acknowledged in our release notes (with permission).

---

Thank you for helping keep `diffml` secure!