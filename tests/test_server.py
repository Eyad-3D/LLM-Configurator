import json
import re
import tempfile
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from unittest.mock import patch

from llm_configurator.server import make_server
from llm_configurator.storage import Store


class ServerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.server = make_server(Store(self.temp.name), port=0, demo=True)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        with urlopen(self.url) as response:
            self.html = response.read().decode()
        self.token = re.search(r'name="session-token" content="([^"]+)"', self.html)[1]

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.temp.cleanup()

    def test_local_ui_and_demo_recommendation(self):
        self.assertIn("Find the right AI", self.html)
        with urlopen(self.url + "/wizard.js") as response:
            self.assertIn("showScreen", response.read().decode())
        with urlopen(self.url + "/selects.js") as response:
            self.assertIn("themed-select", response.read().decode())
        request = Request(self.url + "/api/recommend", data=b'{"context":4096}', headers={"X-Session-Token": self.token})
        with urlopen(request) as response:
            result = json.load(response)
        self.assertTrue(result["demo"])
        self.assertIn("candidates", result)

    def test_refresh_scope_and_busy_worker(self):
        started, release = threading.Event(), threading.Event()
        def refresh_job(*args, **kwargs):
            started.set()
            release.wait(5)
            return {"warnings": []}
        def post(body):
            return urlopen(Request(self.url + "/api/refresh", data=json.dumps(body).encode(),
                                   headers={"X-Session-Token": self.token}))
        with patch("llm_configurator.server.refresh", side_effect=refresh_job) as job:
            try:
                with post({"include_scores": False, "include_models": True}) as response:
                    self.assertEqual(response.status, 202)
                self.assertTrue(started.wait(2))
                with self.assertRaises(HTTPError) as error:
                    post({"include_scores": True, "include_models": False})
                self.assertEqual(error.exception.code, 409)
                self.assertEqual(job.call_args.kwargs, {"include_scores": False, "include_models": True})
            finally:
                release.set()

    def test_credential_endpoints_protect_secret(self):
        from llm_configurator import credentials
        with patch.object(credentials, '_session_key', None), patch.object(credentials, 'vault', side_effect=ValueError('unavailable')):
            request = Request(self.url + '/api/credentials/save', data=b'{"key":"fake-secret","remember":false}', headers={"X-Session-Token": self.token})
            with urlopen(request) as response:
                saved = response.read().decode()
            self.assertNotIn('fake-secret', saved)
            self.assertEqual(json.loads(saved)['source'], 'session')
            with urlopen(Request(self.url + '/api/credentials', headers={"X-Session-Token": self.token})) as response:
                self.assertNotIn('fake-secret', response.read().decode())
            with self.assertRaises(HTTPError) as error:
                urlopen(Request(self.url + '/api/credentials/save', data=b'{"key":"other"}'))
            self.assertEqual(error.exception.code, 403)

    def test_cross_origin_and_missing_session_blocked(self):
        for headers in [{}, {"X-Session-Token": self.token, "Origin": "https://example.com"}]:
            request = Request(self.url + "/api/recommend", data=b"{}", headers=headers)
            with self.assertRaises(HTTPError) as error:
                urlopen(request)
            self.assertEqual(error.exception.code, 403)

    def test_untrusted_host_blocked(self):
        request = Request(self.url, headers={"Host": "attacker.example"})
        with self.assertRaises(HTTPError) as error:
            urlopen(request)
        self.assertEqual(error.exception.code, 403)

    def test_malformed_requirements_return_error(self):
        request = Request(self.url + "/api/recommend", data=b'{"users":0}', headers={"X-Session-Token": self.token})
        with self.assertRaises(HTTPError) as error:
            urlopen(request)
        self.assertEqual(error.exception.code, 400)


if __name__ == "__main__":
    unittest.main()
