import os
from pathlib import Path
from typing import Dict, Optional

class EnvLoader:
    """Centralized environment variable loader that automatically finds and loads .env files."""
    
    def __init__(self):
        self._env_vars: Dict[str, str] = {}
        self._loaded = False
        self._load_env()
    
    def _load_env(self):
        """Load environment variables from .env file"""
        if self._loaded:
            return
            
        # Look for .env files in common locations
        env_paths = [
            Path.cwd() / '.env',
            self._find_project_root() / '.env',
            *[Path.cwd().parents[i] / '.env' for i in range(3)]  # Up to 3 parent levels
        ]
        
        for env_path in env_paths:
            if env_path.exists() and env_path.is_file():
                self._parse_env_file(env_path)
                break
        
        self._loaded = True
    
    def _find_project_root(self) -> Path:
        """Find project root by looking for common indicators"""
        indicators = ['pyproject.toml', 'setup.py', 'requirements.txt', '.git', 'README.md']
        
        for parent in [Path.cwd()] + list(Path.cwd().parents):
            if any((parent / indicator).exists() for indicator in indicators):
                return parent
        
        return Path.cwd()
    
    def _parse_env_file(self, env_path: Path):
        """Parse .env file and load variables"""
        try:
            with open(env_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    
                    # Skip empty lines and comments
                    if not line or line.startswith('#') or '=' not in line:
                        continue
                    
                    # Parse KEY=VALUE format
                    key, value = line.split('=', 1)
                    key, value = key.strip(), value.strip()
                    
                    # Remove quotes if present
                    if (value.startswith('"') and value.endswith('"')) or \
                       (value.startswith("'") and value.endswith("'")):
                        value = value[1:-1]
                    
                    self._env_vars[key] = value
                    os.environ[key] = value  # Also set in os.environ for compatibility
            
            print(f"🔧 Loaded environment variables from: {env_path}")
            
        except Exception as e:
            print(f"Warning: Could not load .env file from {env_path}: {e}")
    
    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Get environment variable value"""
        return self._env_vars.get(key) or os.environ.get(key) or default
    
    def get_api_key(self, service: str) -> Optional[str]:
        """Get API key for a specific service"""
        key_mappings = {
            'openai': ['OPENAI_API_KEY', 'OPENAI_KEY'],
            'gemini': ['GEMINI_API_KEY', 'GOOGLE_API_KEY', 'GEMINI_KEY'],
            'brave': ['BRAVE_API_KEY', 'BRAVE_KEY'],
        }
        
        for key in key_mappings.get(service.lower(), [f"{service.upper()}_API_KEY"]):
            if value := self.get(key):
                return value
        
        return None
    
    def require_api_key(self, service: str) -> str:
        """Get API key for a service, raise error if not found"""
        if not (api_key := self.get_api_key(service)):
            available_keys = list(self._env_vars.keys())
            raise ValueError(
                f"API key for '{service}' not found in environment variables. "
                f"Available keys: {available_keys}. "
                f"Please add one of these to your .env file: {service.upper()}_API_KEY, {service.upper()}_KEY"
            )
        return api_key
    
    def list_available_keys(self) -> Dict[str, str]:
        """List all available environment variables (with API keys masked)"""
        masked_vars = {}
        for key, value in self._env_vars.items():
            if any(word in key.lower() for word in ['key', 'token', 'secret']):
                # Mask sensitive values
                masked_vars[key] = f"{value[:4]}...{value[-4:]}" if len(value) > 8 else "***"
            else:
                masked_vars[key] = value
        return masked_vars

# Global instance
_env_loader = None

def get_env_loader() -> EnvLoader:
    """Get global environment loader instance"""
    global _env_loader
    if _env_loader is None:
        _env_loader = EnvLoader()
    return _env_loader

def get_api_key(service: str) -> Optional[str]:
    """Convenience function to get API key"""
    return get_env_loader().get_api_key(service)

def require_api_key(service: str) -> str:
    """Convenience function to require API key"""
    return get_env_loader().require_api_key(service)
