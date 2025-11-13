"""
SSL Certificate Fix Script
Run this before training if you encounter SSL certificate errors
"""

import os
import ssl

def fix_ssl_issues():
    """Apply SSL fixes for corporate networks with self-signed certificates"""
    
    print("=" * 70)
    print("Applying SSL Certificate Fixes")
    print("=" * 70)
    
    # Method 1: Disable SSL verification globally
    ssl._create_default_https_context = ssl._create_unverified_context
    print("✓ Disabled SSL verification")
    
    # Method 2: Clear certificate bundle environment variables
    os.environ['CURL_CA_BUNDLE'] = ''
    os.environ['REQUESTS_CA_BUNDLE'] = ''
    print("✓ Cleared certificate bundle variables")
    
    # Method 3: Set pip to trust Hugging Face
    print("\nTo permanently fix pip SSL issues, run:")
    print("  pip config set global.trusted-host \"pypi.org files.pythonhosted.org pypi.python.org huggingface.co\"")
    
    print("\n" + "=" * 70)
    print("SSL fixes applied! You can now run: python train.py")
    print("=" * 70)

if __name__ == "__main__":
    fix_ssl_issues()
