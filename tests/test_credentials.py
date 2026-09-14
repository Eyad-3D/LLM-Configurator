import json
import os
import unittest
from unittest.mock import Mock, patch

from llm_configurator import credentials
from llm_configurator.catalogue import fetch_scores, test_connection


class CredentialTests(unittest.TestCase):
    def setUp(self):
        self.session = patch.object(credentials, '_session_key', None)
        self.session.start()
        self.env = patch.dict(os.environ, {}, clear=True)
        self.env.start()
        self.backend = Mock()
        self.backend.get_password.return_value = None
        self.vault = patch.object(credentials, 'vault', return_value=self.backend)
        self.vault.start()

    def tearDown(self):
        self.vault.stop()
        self.env.stop()
        self.session.stop()

    def test_persistent_save_and_status_never_return_secret(self):
        self.backend.get_password.return_value = 'test-secret'
        result = credentials.save('test-secret')
        self.backend.set_password.assert_called_once_with(credentials.SERVICE, credentials.ACCOUNT, 'test-secret')
        self.assertEqual(result, {'configured': True, 'source': 'saved'})
        self.assertNotIn('test-secret', json.dumps(result))

    def test_session_only_overrides_saved_without_writing(self):
        self.backend.get_password.return_value = 'old-saved'
        credentials.save('session-secret', remember=False)
        self.assertEqual(credentials.resolve(), ('session-secret', 'session'))
        self.backend.set_password.assert_not_called()

    def test_failed_save_never_falls_back_to_plaintext_or_session(self):
        self.backend.set_password.side_effect = RuntimeError('secret detail')
        with self.assertRaises(ValueError) as error:
            credentials.save('new-secret')
        self.assertNotIn('secret detail', str(error.exception))
        self.assertIsNone(credentials._session_key)

    def test_remove_clears_saved_and_session(self):
        credentials.save('temporary', remember=False)
        self.backend.get_password.side_effect = ['saved', None]
        self.assertFalse(credentials.remove()['configured'])
        self.backend.delete_password.assert_called_once()
        self.assertIsNone(credentials._session_key)

    def test_environment_still_works(self):
        with patch.dict(os.environ, {'AA_API_KEY': 'environment-secret'}):
            self.assertEqual(credentials.resolve(), ('environment-secret', 'environment'))

    def test_validation_rejects_header_injection(self):
        for key in ['', 'x\ny', 'x y', None]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                credentials.validate_key(key)

    def test_test_connection_does_not_save_key(self):
        with patch('llm_configurator.catalogue.get_json', return_value={'data': []}) as request:
            result = test_connection('candidate-secret')
        self.assertTrue(result['ok'])
        self.assertEqual(request.call_args.args[1], {'x-api-key': 'candidate-secret'})
        self.backend.set_password.assert_not_called()
        self.assertIsNone(credentials._session_key)
        self.assertNotIn('candidate-secret', json.dumps(result))

    def test_refresh_uses_gui_credential(self):
        credentials.save('gui-secret', remember=False)
        with patch('llm_configurator.catalogue.get_json', return_value={'data': [], 'intelligence_index_version': 4.3}) as request:
            fetch_scores()
        self.assertEqual(request.call_args.args[1]['x-api-key'], 'gui-secret')
