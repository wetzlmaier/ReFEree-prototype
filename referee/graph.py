"""
Project Context Graph Construction for ReFEree.

Builds a heterogeneous directed acyclic graph (DAG) representing code entities
(functions, classes, variables, imports) and their dependency relations using
5 relation types (assign, as, refers, typeof, inherits) based on AST analysis
as defined in Section 3.2 and Appendix A.2 of the ReFEree paper.
"""

from __future__ import annotations
import ast
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any


def _empty_metadata() -> Dict[str, Any]:
    return {}


class EdgeRelation(str, Enum):
    """The 5 relation types for variable and dependency flow (Cheng et al., 2024)."""
    ASSIGN = "assign"      # Correspondence in assignment statement (creation/mutation)
    AS = "as"              # With or except statement binding
    REFERS = "refers"      # Reference to an existing variable, attribute, or function call
    TYPEOF = "typeof"      # Data type of return value or variable
    INHERITS = "inherits"  # Class inheritance relation


@dataclass
class CodeEntity:
    """Represents a code entity node in the project context graph."""
    id: str
    name: str
    ast_type: str  # 'function_definition', 'class_definition', 'assignment', 'import_statement'
    file_path: str
    code_context: str
    docstring: Optional[str] = None
    scope: str = ""
    metadata: Dict[str, Any] = field(default_factory=_empty_metadata)

    def __repr__(self) -> str:
        return f"<CodeEntity id={self.id!r} type={self.ast_type!r} name={self.name!r}>"


class ProjectContextGraph:
    """
    Heterogeneous directed acyclic graph G = {(h, r, t) | h, t in E, r in R}.
    """

    def __init__(self, root_dir: Optional[str] = None):
        self.root_dir = os.path.abspath(root_dir) if root_dir else None
        self.entities: Dict[str, CodeEntity] = {}
        self.edges: List[Tuple[str, EdgeRelation, str]] = []
        self.outgoing: Dict[str, List[Tuple[EdgeRelation, str]]] = {}
        self.incoming: Dict[str, List[Tuple[EdgeRelation, str]]] = {}
        # Mapping from (file_path, name) to entity ID for fast lookup
        self.name_to_entity_id: Dict[Tuple[str, str], str] = {}
        # Global name map for top-level entities (by entity name)
        self.top_level_names: Dict[str, List[str]] = {}

    def add_entity(self, entity: CodeEntity) -> None:
        self.entities[entity.id] = entity
        self.name_to_entity_id[(entity.file_path, entity.name)] = entity.id
        if entity.scope in ("", "<module>"):
            self.top_level_names.setdefault(entity.name, []).append(entity.id)
        if entity.id not in self.outgoing:
            self.outgoing[entity.id] = []
        if entity.id not in self.incoming:
            self.incoming[entity.id] = []

    def add_edge(self, head_id: str, relation: EdgeRelation, tail_id: str) -> None:
        edge = (head_id, relation, tail_id)
        if edge not in self.edges:
            self.edges.append(edge)
            self.outgoing.setdefault(head_id, []).append((relation, tail_id))
            self.incoming.setdefault(tail_id, []).append((relation, head_id))

    def get_entity(self, entity_id: str) -> Optional[CodeEntity]:
        return self.entities.get(entity_id)

    def find_entity_by_name(self, name: str, file_path: Optional[str] = None) -> Optional[CodeEntity]:
        if file_path and (file_path, name) in self.name_to_entity_id:
            return self.entities[self.name_to_entity_id[(file_path, name)]]
        if name in self.top_level_names and self.top_level_names[name]:
            return self.entities[self.top_level_names[name][0]]
        return None

    def add_python_source(self, source_code: str, file_path: str = "main.py") -> List[CodeEntity]:
        """Parses a Python source string, creates entity nodes and dependency edges."""
        norm_file = os.path.normpath(file_path)
        try:
            tree = ast.parse(source_code)
        except SyntaxError:
            # If partial code snippet, wrap or handle gracefully
            tree = ast.parse("")

        lines = source_code.splitlines(keepends=True)

        def get_source_segment(node: ast.AST) -> str:
            if hasattr(ast, "get_source_segment") and source_code:
                seg = ast.get_source_segment(source_code, node)
                if seg:
                    return seg
            lineno = getattr(node, "lineno", None)
            end_lineno = getattr(node, "end_lineno", None)
            if isinstance(lineno, int) and isinstance(end_lineno, int):
                start = lineno - 1
                end = end_lineno
                return "".join(lines[start:end])
            return ""

        added_entities: List[CodeEntity] = []
        # First pass: collect entities
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                ent_id = f"{norm_file}:{node.name}"
                scope = getattr(node, "_parent_scope", "<module>")
                docstring = ast.get_docstring(node)
                code_context = get_source_segment(node)
                ret_type = ""
                if node.returns:
                    ret_type = ast.unparse(node.returns) if hasattr(ast, "unparse") else ""

                entity = CodeEntity(
                    id=ent_id,
                    name=node.name,
                    ast_type="function_definition",
                    file_path=norm_file,
                    code_context=code_context,
                    docstring=docstring,
                    scope=scope,
                    metadata={
                        "args": [arg.arg for arg in node.args.args],
                        "return_type": ret_type,
                        "lineno": node.lineno,
                    },
                )
                self.add_entity(entity)
                added_entities.append(entity)

            elif isinstance(node, ast.ClassDef):
                ent_id = f"{norm_file}:{node.name}"
                scope = getattr(node, "_parent_scope", "<module>")
                docstring = ast.get_docstring(node)
                code_context = get_source_segment(node)
                bases = [ast.unparse(b) for b in node.bases] if hasattr(ast, "unparse") else []

                entity = CodeEntity(
                    id=ent_id,
                    name=node.name,
                    ast_type="class_definition",
                    file_path=norm_file,
                    code_context=code_context,
                    docstring=docstring,
                    scope=scope,
                    metadata={"bases": bases, "lineno": node.lineno},
                )
                self.add_entity(entity)
                added_entities.append(entity)

            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                target_names: List[str] = []
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            target_names.append(target.id)
                        elif isinstance(target, ast.Attribute):
                            target_names.append(target.attr)
                else:
                    if isinstance(node.target, ast.Name):
                        target_names.append(node.target.id)
                    elif isinstance(node.target, ast.Attribute):
                        target_names.append(node.target.attr)

                code_context = get_source_segment(node)
                for t_name in target_names:
                    ent_id = f"{norm_file}:assign:{t_name}:{node.lineno}"
                    entity = CodeEntity(
                        id=ent_id,
                        name=t_name,
                        ast_type="assignment",
                        file_path=norm_file,
                        code_context=code_context,
                        metadata={"lineno": node.lineno},
                    )
                    self.add_entity(entity)
                    added_entities.append(entity)

            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                code_context = get_source_segment(node)
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        ent_name = alias.asname or alias.name
                        ent_id = f"{norm_file}:import:{ent_name}"
                        entity = CodeEntity(
                            id=ent_id,
                            name=ent_name,
                            ast_type="import_statement",
                            file_path=norm_file,
                            code_context=code_context,
                            metadata={
                                "module": alias.name,
                                "name": alias.name,
                                "asname": alias.asname,
                                "lineno": node.lineno,
                            },
                        )
                        self.add_entity(entity)
                        added_entities.append(entity)
                else:
                    module = node.module or ""
                    for alias in node.names:
                        ent_name = alias.asname or alias.name
                        ent_id = f"{norm_file}:import:{ent_name}"
                        entity = CodeEntity(
                            id=ent_id,
                            name=ent_name,
                            ast_type="import_statement",
                            file_path=norm_file,
                            code_context=code_context,
                            metadata={
                                "module": module,
                                "name": alias.name,
                                "asname": alias.asname,
                                "lineno": node.lineno,
                            },
                        )
                        self.add_entity(entity)
                        added_entities.append(entity)

        # Second pass: establish edges (5 relation types: assign, as, refers, typeof, inherits)
        for entity in added_entities:
            # We locate the ast node for this entity if possible
            if entity.ast_type == "class_definition":
                for base_name in entity.metadata.get("bases", []):
                    # Edge: class inherits base
                    target_ent = self.find_entity_by_name(base_name, norm_file)
                    target_id = target_ent.id if target_ent else f"external:{base_name}"
                    self.add_edge(entity.id, EdgeRelation.INHERITS, target_id)

            elif entity.ast_type == "function_definition":
                ret_type = entity.metadata.get("return_type")
                if ret_type:
                    self.add_edge(entity.id, EdgeRelation.TYPEOF, f"type:{ret_type}")

        # Scan detailed relationships inside each function / block
        class DependencyVisitor(ast.NodeVisitor):
            def __init__(self, graph: ProjectContextGraph, file_p: str):
                self.graph = graph
                self.file_p = file_p
                self.current_scope_id: Optional[str] = None

            def visit_FunctionDef(self, node: ast.FunctionDef):
                prev = self.current_scope_id
                self.current_scope_id = f"{self.file_p}:{node.name}"
                self.generic_visit(node)
                self.current_scope_id = prev

            def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
                prev = self.current_scope_id
                self.current_scope_id = f"{self.file_p}:{node.name}"
                self.generic_visit(node)
                self.current_scope_id = prev

            def visit_Assign(self, node: ast.Assign):
                if self.current_scope_id:
                    for target in node.targets:
                        t_name = ""
                        if isinstance(target, ast.Name):
                            t_name = target.id
                        elif isinstance(target, ast.Attribute):
                            t_name = target.attr
                        if t_name:
                            assign_id = f"{self.file_p}:assign:{t_name}:{node.lineno}"
                            self.graph.add_edge(self.current_scope_id, EdgeRelation.ASSIGN, assign_id)
                self.generic_visit(node)

            def visit_With(self, node: ast.With):
                if self.current_scope_id:
                    for item in node.items:
                        if item.optional_vars and isinstance(item.optional_vars, ast.Name):
                            target_name = item.optional_vars.id
                            expr_repr = ast.unparse(item.context_expr) if hasattr(ast, "unparse") else "with_expr"
                            var_id = f"{self.file_p}:var:{target_name}:{node.lineno}"
                            if var_id not in self.graph.entities:
                                self.graph.add_entity(CodeEntity(
                                    id=var_id,
                                    name=target_name,
                                    ast_type="variable",
                                    file_path=self.file_p,
                                    code_context=f"with {expr_repr} as {target_name}:",
                                ))
                            self.graph.add_edge(self.current_scope_id, EdgeRelation.AS, var_id)
                self.generic_visit(node)

            def visit_ExceptHandler(self, node: ast.ExceptHandler):
                if self.current_scope_id and node.name:
                    type_str = ast.unparse(node.type) if node.type and hasattr(ast, "unparse") else "Exception"
                    var_id = f"{self.file_p}:var:{node.name}:{node.lineno}"
                    if var_id not in self.graph.entities:
                        self.graph.add_entity(CodeEntity(
                            id=var_id,
                            name=node.name,
                            ast_type="variable",
                            file_path=self.file_p,
                            code_context=f"except {type_str} as {node.name}:",
                        ))
                    self.graph.add_edge(self.current_scope_id, EdgeRelation.AS, var_id)
                self.generic_visit(node)

            def visit_Call(self, node: ast.Call):
                if self.current_scope_id:
                    called_name = None
                    if isinstance(node.func, ast.Name):
                        called_name = node.func.id
                    elif isinstance(node.func, ast.Attribute):
                        called_name = node.func.attr
                        # Attribute chain like itertools.chain or urllib.parse.urlsplit
                        chain = self._get_attribute_chain(node.func)
                        if chain:
                            base_name = chain.split(".")[0]
                            base_target = self.graph.find_entity_by_name(base_name, self.file_p)
                            if base_target and base_target.ast_type == "import_statement":
                                mod = base_target.metadata.get("module", base_name)
                                attr_part = ".".join(chain.split(".")[1:])
                                ent_id = f"{self.file_p}:import_attr:{chain}"
                                if ent_id not in self.graph.entities:
                                    self.graph.add_entity(CodeEntity(
                                        id=ent_id,
                                        name=chain,
                                        ast_type="import_statement",
                                        file_path=self.file_p,
                                        code_context=chain,
                                        metadata={
                                            "module": mod,
                                            "name": attr_part,
                                        },
                                    ))
                                self.graph.add_edge(self.current_scope_id, EdgeRelation.REFERS, ent_id)
                            elif base_target:
                                self.graph.add_edge(self.current_scope_id, EdgeRelation.REFERS, base_target.id)
                            else:
                                self.graph.add_edge(self.current_scope_id, EdgeRelation.REFERS, f"import_or_external:{chain}")

                    if called_name:
                        target = self.graph.find_entity_by_name(called_name, self.file_p)
                        target_id = target.id if target else f"external:{called_name}"
                        self.graph.add_edge(self.current_scope_id, EdgeRelation.REFERS, target_id)
                self.generic_visit(node)

            def visit_Name(self, node: ast.Name):
                if self.current_scope_id and isinstance(node.ctx, ast.Load):
                    name = node.id
                    target = self.graph.find_entity_by_name(name, self.file_p)
                    if target and target.id != self.current_scope_id:
                        self.graph.add_edge(self.current_scope_id, EdgeRelation.REFERS, target.id)
                self.generic_visit(node)

            def _get_attribute_chain(self, node: ast.AST) -> str:
                parts: List[str] = []
                curr = node
                while isinstance(curr, ast.Attribute):
                    parts.append(curr.attr)
                    curr = curr.value
                if isinstance(curr, ast.Name):
                    parts.append(curr.id)
                    return ".".join(reversed(parts))
                return ""

        visitor = DependencyVisitor(self, norm_file)
        visitor.visit(tree)
        return added_entities

    def add_python_file(self, file_path: str) -> List[CodeEntity]:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        return self.add_python_source(content, file_path=file_path)

    def add_repository(self, repo_dir: str) -> None:
        """Indexes all Python files in a directory/repository."""
        self.root_dir = os.path.abspath(repo_dir)
        for root, _, files in os.walk(repo_dir):
            for file in files:
                if file.endswith(".py"):
                    full_path = os.path.join(root, file)
                    try:
                        self.add_python_file(full_path)
                    except Exception:
                        pass
