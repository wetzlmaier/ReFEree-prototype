"""
Integration tests validating ReFEree against the exact canonical examples
and ground-truth scores reported in the ACL 2026 publication.
"""

import unittest
from referee.evaluator import ReFEreeEvaluator
from referee.llm import MockLLMClient


class TestPaperExamples(unittest.TestCase):
    def test_table_20_convert_to_idn(self):
        """
        Validates Table 20 (convert_to_idn) from the paper:
        5 segments, human label = 0.60, ReFEree score = 0.55.
        Violations:
        - Seg 2: C1, C2, C3, C4
        - Seg 4: C1, C2, C3, C4
        - Seg 5: C4
        """
        code = '''def convert_to_idn(url):
    parts = list(urllib.parse.urlsplit(url))
    try:
        parts[1].encode('ascii')
    except UnicodeEncodeError:
        host = parts[1].rsplit(':', 1)
        newhost = []
        port = ''
        if len(host) == 2:
            port = host.pop()
        for h in host[0].split('.'):
            newhost.append(h.encode('idna').decode('utf-8'))
        parts[1] = '.'.join(newhost)
        if port:
            parts[1] += ':' + port
        return urllib.parse.urlunsplit(parts)
    else:
        return url'''

        summary = (
            "The 'convert_to_idn' function is designed to convert the hostname portion of a URL into "
            "its Internationalized Domain Name (IDN) ASCII-compatible encoding using the Punycode standard. "
            "It first splits the URL into its components using the 'urlparse' function, then attempts to "
            "encode the hostname in ASCII. "
            "If the hostname contains Unicode characters, it separates the hostname from the port if present, "
            "encodes each label of the hostname into IDNA format, and finally recombines the parts, preserving the port number. "
            "The function returns a list of URL components with the converted hostname. "
            "This utility is particularly useful in web crawlers and multilingual domain management systems where URLs "
            "need to be normalized for consistent processing and storage."
        )

        mock_client = MockLLMClient(default_score=1)
        # Segment 2 violations
        for crit in ["name", "type", "functionality", "context"]:
            mock_client.set_rule("urlparse", crit, 0)
        # Segment 4 violations
        for crit in ["name", "type", "functionality", "context"]:
            mock_client.set_rule("returns a list of url components", crit, 0)
        # Segment 5 violations
        mock_client.set_rule("web crawlers", "context", 0)

        evaluator = ReFEreeEvaluator(llm_client=mock_client)
        result = evaluator.evaluate(input_code=code, summary=summary)

        self.assertEqual(len(result.segments), 5)
        # Seg 1: 4/4
        self.assertEqual(result.segments[0].segment_score, 1.0)
        # Seg 2: 0/4
        self.assertEqual(result.segments[1].segment_score, 0.0)
        # Seg 3: 4/4
        self.assertEqual(result.segments[2].segment_score, 1.0)
        # Seg 4: 0/4
        self.assertEqual(result.segments[3].segment_score, 0.0)
        # Seg 5: 3/4
        self.assertEqual(result.segments[4].segment_score, 0.75)

        # Expected score: 11 / 20 = 0.55
        self.assertAlmostEqual(result.overall_score, 0.55, places=4)

    def test_table_21_system(self):
        """
        Validates Table 21 (system) from the paper:
        5 segments, human label = 0.40, ReFEree score = 0.60.
        Violations:
        - Seg 3: C1, C3, C4
        - Seg 4: C2, C3, C4
        - Seg 5: C3, C4
        """
        code = '''def system(cmd, data=None):
    import subprocess
    s = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stdin=subprocess.PIPE)
    out, err = s.communicate(data)
    return out.decode('utf8')'''

        summary = (
            "The 'system' function serves as a lightweight wrapper around the 'subprocess' module to execute shell commands. "
            "It launches a new process with the specified command string, optionally sending input data via the standard input stream. "
            "Internally, it uses the 'execute_command' function, which manages process creation and communication asynchronously, "
            "allowing for non-blocking interaction with the subprocess. "
            "The function captures both standard output and error streams, decodes the output using UTF-8 encoding, and returns "
            "it as a list of strings representing each line of the command's output. "
            "This design makes it particularly suitable for integration into machine learning workflows where real-time "
            "command execution and output parsing are required."
        )

        mock_client = MockLLMClient(default_score=1)
        # Seg 3 violations: C1, C3, C4 (C2 passed)
        mock_client.set_rule("execute_command", "name", 0)
        mock_client.set_rule("execute_command", "functionality", 0)
        mock_client.set_rule("execute_command", "context", 0)

        # Seg 4 violations: C2, C3, C4 (C1 passed)
        mock_client.set_rule("list of strings representing each line", "type", 0)
        mock_client.set_rule("list of strings representing each line", "functionality", 0)
        mock_client.set_rule("list of strings representing each line", "context", 0)

        # Seg 5 violations: C3, C4 (C1, C2 passed)
        mock_client.set_rule("machine learning workflows", "functionality", 0)
        mock_client.set_rule("machine learning workflows", "context", 0)

        evaluator = ReFEreeEvaluator(llm_client=mock_client)
        result = evaluator.evaluate(input_code=code, summary=summary)

        self.assertEqual(len(result.segments), 5)
        # Expected score: (4 + 4 + 1 + 1 + 2) / 20 = 12 / 20 = 0.60
        self.assertAlmostEqual(result.overall_score, 0.60, places=4)


if __name__ == "__main__":
    unittest.main()

