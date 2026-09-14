"""API credentials in the OS vault or explicit process-only storage; never SQLite."""
import os
import threading

SERVICE = 'LLMConfigurator'
ACCOUNT = 'artificial-analysis'
_LOCK = threading.RLock()
_session_key = None


def validate_key(value):
    if not isinstance(value, str):
        raise ValueError('Enter an API key.')
    value = value.strip()
    if not value or len(value) > 4096 or any(ord(c) < 33 or ord(c) > 126 for c in value):
        raise ValueError('Enter a non-empty API key without spaces or control characters.')
    return value


def vault():
    # Select only native credential stores, never a third-party plaintext backend.
    try:
        import keyring
        backend = keyring.get_keyring()
        candidates = getattr(backend, 'backends', [backend])
        for candidate in candidates:
            if type(candidate).__module__ in {'keyring.backends.Windows', 'keyring.backends.macOS',
                                              'keyring.backends.SecretService', 'keyring.backends.kwallet'}:
                if candidate.priority > 0:
                    return candidate
    except Exception:
        pass
    raise ValueError('Secure credential storage is unavailable. Unlock your OS credential store or uncheck Remember on this computer to use the key for this session only.')


def resolve():
    with _LOCK:
        if _session_key:
            return _session_key, 'session'
        try:
            saved = vault().get_password(SERVICE, ACCOUNT)
            if saved:
                return saved, 'saved'
        except Exception:
            pass
        return (os.environ.get('AA_API_KEY') or None), 'environment' if os.environ.get('AA_API_KEY') else 'none'


def status():
    key, source = resolve()
    return {'configured': bool(key), 'source': source}


def save(value, remember=True):
    global _session_key
    value = validate_key(value)
    if type(remember) is not bool:
        raise ValueError('Remember must be a boolean.')
    with _LOCK:
        if remember:
            try:
                vault().set_password(SERVICE, ACCOUNT, value)
            except Exception:
                raise ValueError('Could not save to the OS credential store. Unlock it or uncheck Remember on this computer for session-only use.') from None
            _session_key = None
        else:
            _session_key = value
        return status()


def remove():
    global _session_key
    with _LOCK:
        try:
            backend = vault()
        except ValueError:
            if _session_key is None:
                raise ValueError('Cannot access the OS credential store to remove a saved key.') from None
            backend = None
        if backend is not None:
            try:
                if backend.get_password(SERVICE, ACCOUNT) is not None:
                    backend.delete_password(SERVICE, ACCOUNT)
            except Exception:
                raise ValueError('Could not remove the saved key. Unlock your OS credential store and try again.') from None
        _session_key = None
        return status()
