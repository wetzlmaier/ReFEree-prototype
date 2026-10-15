"""
Unit tests for Code-Related Information Searching (1-hop DFS).
"""

import unittest
from referee.graph import ProjectContextGraph
from referee.search import CodeRelatedInfoSearcher
from referee.external_docs import ExternalDocRegistry


class TestSearch(unittest.TestCase):
    def test_internal_dependency_search(self):
        code = '''
def helper(x):
    return x * 2

def main_func(val):
    result = helper(val)
    return result
'''
        graph = ProjectContextGraph()
        searcher = CodeRelatedInfoSearcher(graph=graph)
        items = searcher.search_for_code(code, target_entity_name="main_func")

        names = [item.name for item in items]
        self.assertIn("helper", names)
        # Verify content contains code body
        helper_item = next(i for i in items if i.name == "helper")
        self.assertIn("def helper(x):", helper_item.content)
        self.assertEqual(helper_item.source_type, "internal")

    def test_external_dependency_search(self):
        code = '''
import datetime

def calc_days(to_date, from_date):
    diff = datetime.timedelta(days=1)
    return to_date - diff
'''
        graph = ProjectContextGraph()
        registry = ExternalDocRegistry()
        searcher = CodeRelatedInfoSearcher(graph=graph, external_docs=registry)
        items = searcher.search_for_code(code, target_entity_name="calc_days")

        names = [item.name for item in items]
        # Should have found datetime.timedelta or datetime
        found = any("timedelta" in n or "datetime" in n for n in names)
        self.assertTrue(found)

    def test_cross_file_dependency_search(self):
        graph = ProjectContextGraph()
        # Add external module file into graph
        utils_code = '''
def key_prefix(prefix, date, account_id, region):
    return f"{prefix}/{account_id}/{region}/{date.year}"
'''
        graph.add_python_source(utils_code, file_path="utils/collection_utils.py")

        main_code = '''
from utils.collection_utils import key_prefix

def s3_key_prefixes(prefix, org_ids):
    return [key_prefix(prefix, 'day', 'acc', 'reg')]
'''
        searcher = CodeRelatedInfoSearcher(graph=graph)
        items = searcher.search_for_code(main_code, file_path="main.py", target_entity_name="s3_key_prefixes")

        names = [item.name for item in items]
        self.assertIn("key_prefix", names)
        item = next(i for i in items if i.name == "key_prefix")
        self.assertEqual(item.source_type, "cross-file")
        self.assertIn("def key_prefix", item.content)

    def test_format_paper_style(self):
        code = '''
def time_interval(to_date, from_date):
    return to_date - from_date

def compute():
    delta = time_interval(d2, d1)
    return delta
'''
        graph = ProjectContextGraph()
        searcher = CodeRelatedInfoSearcher(graph=graph)
        items = searcher.search_for_code(code, target_entity_name="compute")
        formatted = searcher.format_related_information(items)

        self.assertIn("# time_interval #", formatted)
        self.assertIn("def time_interval(to_date, from_date):", formatted)


if __name__ == "__main__":
    unittest.main()
