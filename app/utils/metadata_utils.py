# En app/utils/metadata_utils.py
SECURITY_DOMAINS = {
    "cloud": ["azure", "aws", "gcp", "cloud", "s3", "blob storage"],
    "owasp": ["owasp", "top 10", "web vuln", "xss", "sqli", "csrf"],
    "ad": ["active directory", "kerberos", "ldap", "domain", "golden ticket"],
    "binary": ["reverse", "ida pro", "ghidra", "binary analysis"],
    "mobile": ["android", "ios", "mobile security"],
    "network": ["packet", "tcp/ip", "protocol", "wireshark"],
    "exploit": ["rop", "shellcode", "buffer overflow", "exploit dev"]
}

def detect_security_domain(title: str, content: str) -> list:
    domains = []
    title_lower = title.lower()
    content_lower = content[:1000].lower()  # Muestra inicial
    
    for domain, keywords in SECURITY_DOMAINS.items():
        if any(kw in title_lower for kw in keywords) or any(kw in content_lower for kw in keywords):
            domains.append(domain)
    return domains if domains else ["general"]