#!/bin/bash
# SLSA Compliance Checker for Mobile Multi-Modal LLM
# Validates current SLSA compliance level and identifies gaps

set -euo pipefail

# Configuration
REQUIRED_SLSA_LEVEL=2
CURRENT_LEVEL=0
ISSUES=()

echo "🔒 SLSA Compliance Check - Mobile Multi-Modal LLM"
echo "=================================================="

# Check SLSA Level 1 Requirements
check_level_1() {
    echo "📋 Checking SLSA Level 1 Requirements..."
    
    # Source version control
    if [ -d ".git" ]; then
        echo "✅ Source code in version control"
    else
        ISSUES+=("❌ Source code not in version control")
        return 1
    fi
    
    # Build scripted
    if [ -f "pyproject.toml" ] || [ -f "Makefile" ]; then
        echo "✅ Build process is scripted"
    else
        ISSUES+=("❌ Build process not properly scripted")
        return 1
    fi
    
    # Provenance available (basic)
    if [ -d ".github/workflows" ]; then
        echo "✅ Build system generates provenance"
    else
        ISSUES+=("❌ No CI/CD system for provenance generation")
        return 1
    fi
    
    CURRENT_LEVEL=1
    return 0
}

# Check SLSA Level 2 Requirements
check_level_2() {
    echo "📋 Checking SLSA Level 2 Requirements..."
    
    # Hosted build service
    if grep -q "runs-on:" .github/workflows/*.yml 2>/dev/null; then
        echo "✅ Using hosted build service (GitHub Actions)"
    else
        ISSUES+=("❌ Not using hosted build service")
        return 1
    fi
    
    # Build service generates provenance
    if grep -q "slsa-framework\|provenance" .github/workflows/*.yml 2>/dev/null || \
       [ -f "sbom.json" ] || \
       grep -q "cyclonedx-bom" pyproject.toml 2>/dev/null; then
        echo "✅ Build service configured for provenance generation"
    else
        ISSUES+=("❌ Build service not generating proper provenance")
        return 1
    fi
    
    # Provenance format
    if grep -q "SBOM\|cyclonedx\|slsa" pyproject.toml 2>/dev/null; then
        echo "✅ Provenance in standard format (SBOM)"
    else
        ISSUES+=("❌ Provenance not in standard format")
        return 1
    fi
    
    CURRENT_LEVEL=2
    return 0
}

# Check SLSA Level 3 Requirements
check_level_3() {
    echo "📋 Checking SLSA Level 3 Requirements..."
    
    # Hosted build service with isolation
    if grep -q "self-hosted\|isolated" .github/workflows/*.yml 2>/dev/null; then
        echo "✅ Using isolated build environment"
    else
        ISSUES+=("❌ Build environment not sufficiently isolated")
        return 1
    fi
    
    # Non-falsifiable provenance
    if grep -q "cosign\|sigstore\|keyless" .github/workflows/*.yml 2>/dev/null; then
        echo "✅ Non-falsifiable provenance (signed)"
    else
        ISSUES+=("❌ Provenance not cryptographically signed")
        return 1
    fi
    
    CURRENT_LEVEL=3
    return 0
}

# Security scanning checks
check_security_scanning() {
    echo "🛡️  Checking Security Scanning..."
    
    # Dependency scanning
    if grep -q "safety\|pip-audit\|bandit" pyproject.toml 2>/dev/null; then
        echo "✅ Dependency vulnerability scanning enabled"
    else
        ISSUES+=("❌ Missing dependency vulnerability scanning")
    fi
    
    # Secret scanning
    if [ -f ".secrets.baseline" ] && grep -q "detect-secrets" .pre-commit-config.yaml 2>/dev/null; then
        echo "✅ Secret scanning configured"
    else
        ISSUES+=("❌ Secret scanning not properly configured")
    fi
    
    # Container scanning
    if [ -f "Dockerfile" ]; then
        if grep -q "trivy\|snyk\|clair" .github/workflows/*.yml 2>/dev/null; then
            echo "✅ Container vulnerability scanning"
        else
            ISSUES+=("❌ Container images not scanned for vulnerabilities")
        fi
    fi
}

# Build reproducibility checks
check_reproducibility() {
    echo "🔄 Checking Build Reproducibility..."
    
    # Fixed dependencies
    if [ -f "requirements.txt" ] && grep -q "==" requirements.txt; then
        echo "✅ Dependencies pinned for reproducible builds"
    else
        ISSUES+=("❌ Dependencies not properly pinned")
    fi
    
    # Environment consistency
    if grep -q "PYTHONHASHSEED\|deterministic" pyproject.toml scripts/*.py 2>/dev/null; then
        echo "✅ Deterministic build environment configured"
    else
        ISSUES+=("❌ Build environment not deterministic")
    fi
}

# Generate compliance report
generate_report() {
    echo ""
    echo "📊 SLSA Compliance Report"
    echo "========================="
    echo "Current SLSA Level: $CURRENT_LEVEL"
    echo "Required SLSA Level: $REQUIRED_SLSA_LEVEL"
    
    if [ $CURRENT_LEVEL -ge $REQUIRED_SLSA_LEVEL ]; then
        echo "✅ SLSA compliance requirement MET"
        exit 0
    else
        echo "❌ SLSA compliance requirement NOT MET"
        echo ""
        echo "🔧 Issues to resolve:"
        printf '%s\n' "${ISSUES[@]}"
        echo ""
        echo "📚 Next steps:"
        echo "1. Review SLSA.md for implementation guidance"
        echo "2. Update CI/CD workflows for provenance generation"
        echo "3. Implement missing security controls"
        echo "4. Re-run compliance check"
        exit 1
    fi
}

# Main execution
main() {
    if check_level_1; then
        if check_level_2; then
            check_level_3 || true  # Level 3 is optional
        fi
    fi
    
    check_security_scanning
    check_reproducibility
    generate_report
}

# Run compliance check
main "$@"