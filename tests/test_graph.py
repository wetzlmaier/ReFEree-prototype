"""
Unit tests for Project Context Graph construction and 5 edge relations.
"""

import unittest
from referee.graph import ProjectContextGraph, EdgeRelation


class TestProjectContextGraph(unittest.TestCase):
    def setUp(self):
        self.graph = ProjectContextGraph()

    def test_parse_entities(self):
        code = '''
from utils import key_prefix

class PrefixManager(BaseManager):
    """Manages prefixes."""
    pass

def s3_key_prefixes(prefix: str, ids: list) -> list:
    """Generate key prefixes."""
    delta = time_interval(to_date, from_date)
    result = [key_prefix(prefix, ids)]
    return result
'''
        entities = self.graph.add_python_source(code, file_path="main.py")
        ent_names = {e.name: e for e in entities}

        # Check entity types
        self.assertIn("key_prefix", ent_names)
        self.assertEqual(ent_names["key_prefix"].ast_type, "import_statement")

        self.assertIn("PrefixManager", ent_names)
        self.assertEqual(ent_names["PrefixManager"].ast_type, "class_definition")
        self.assertEqual(ent_names["PrefixManager"].docstring, "Manages prefixes.")

        self.assertIn("s3_key_prefixes", ent_names)
        self.assertEqual(ent_names["s3_key_prefixes"].ast_type, "function_definition")

        # Check assignment entity
        assigns = [e for e in entities if e.ast_type == "assignment"]
        assign_names = [e.name for e in assigns]
        self.assertIn("delta", assign_names)

    def test_five_relation_types(self):
        code = '''
class Animal:
    pass

class Dog(Animal):
    """Dog class."""
    pass

def process() -> str:
    x = 10
    with open("file.txt") as f:
        pass
    try:
        val = call_helper()
    except ValueError as err:
        pass
    return "done"
'''
        self.graph.add_python_source(code, file_path="example.py")
        relations = {edge[1] for edge in self.graph.edges}

        # 1. inherits relation
        self.assertIn(EdgeRelation.INHERITS, relations)

        # 2. typeof relation
        self.assertIn(EdgeRelation.TYPEOF, relations)

        # 3. assign relation
        self.assertIn(EdgeRelation.ASSIGN, relations)

        # 4. as relation (with or except)
        self.assertIn(EdgeRelation.AS, relations)

        # 5. refers relation (call_helper, open, etc.)
        self.assertIn(EdgeRelation.REFERS, relations)


if __name__ == "__main__":
    unittest.main()

