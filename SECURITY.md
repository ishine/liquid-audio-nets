# Security Policy

## Supported Versions

We release patches for security vulnerabilities. Which versions are eligible for receiving such patches depends on the CVSS v3.0 Rating:

| CVSS v3.0 | Supported Versions                        |
| --------- | ----------------------------------------- |
| 9.0-10.0  | Releases within the previous three months |
| 4.0-8.9   | Most recent release                       |

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, please report them by email to: **security@terragon.dev**

You should receive a response within 48 hours. If for some reason you do not, please follow up via email to ensure we received your original message.

### What to Include

Please include the requested information listed below (as much as you can provide) to help us better understand the nature and scope of the possible issue:

* Type of issue (e.g. buffer overflow, SQL injection, cross-site scripting, etc.)
* Full paths of source file(s) related to the manifestation of the issue
* The location of the affected source code (tag/branch/commit or direct URL)
* Any special configuration required to reproduce the issue
* Step-by-step instructions to reproduce the issue
* Proof-of-concept or exploit code (if possible)
* Impact of the issue, including how an attacker might exploit the issue

### Preferred Languages

We prefer all communications to be in English.

## Security Considerations for Embedded Systems

### Hardware Security

- **Secure Boot**: Implement secure boot mechanisms where possible
- **Memory Protection**: Use MPU/MMU features when available
- **Debug Interface**: Disable debug interfaces in production builds
- **Side-Channel Attacks**: Consider power analysis and timing attack mitigations

### Software Security

- **Input Validation**: All audio inputs and configuration parameters are validated
- **Buffer Overflow Protection**: Stack canaries and bounds checking where possible
- **Cryptographic Libraries**: Use vetted crypto libraries, not custom implementations
- **Firmware Updates**: Implement secure update mechanisms with signature verification

### Model Security

- **Model Integrity**: Verify model signatures before loading
- **Model Encryption**: Consider encrypting models containing proprietary algorithms
- **Adversarial Robustness**: Test against adversarial audio inputs
- **Data Privacy**: Ensure on-device processing preserves user privacy

## Security Testing

We employ several security testing methods:

- **Static Analysis**: Code is analyzed using multiple static analysis tools
- **Dynamic Analysis**: Runtime security testing including fuzzing
- **Hardware Testing**: Security testing on target embedded platforms
- **Third-party Audits**: Periodic security audits by external experts

### Running Security Checks

```bash
# Run security check script
python scripts/security_check.py

# Scan for secrets
detect-secrets scan --all-files --baseline .secrets.baseline

# Check dependencies for known vulnerabilities
cargo audit  # For Rust dependencies
pip-audit    # For Python dependencies
```

## Security Best Practices for Contributors

### Code Review

- All code changes require security-focused review
- Pay special attention to memory management in C/C++ code
- Review Rust unsafe blocks carefully
- Validate all input parsing and deserialization code

### Dependency Management

- Keep dependencies up to date
- Review security advisories for all dependencies
- Use dependency pinning in production
- Minimize the dependency footprint

### Embedded Development

- Use compiler security features (stack canaries, FORTIFY_SOURCE)
- Enable all relevant compiler warnings
- Use memory-safe alternatives where possible
- Test on resource-constrained environments

## Coordinated Disclosure Timeline

We ask that you give us up to 90 days to investigate and mitigate an issue you report. After that time period, we welcome any public disclosure of the issue.

## Comments on this Policy

If you have suggestions on how this process could be improved, please submit a pull request or send an email to security@terragon.dev.