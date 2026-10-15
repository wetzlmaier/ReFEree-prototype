"""
Code-Related Information Searching (1-hop DFS and dependency selection).

Implements the two-step static program analysis approach described in Section 3.2
and Appendix A.2 of the ReFEree paper:
1. Traverse 1-hop dependencies from the input code entities via DFS.
2. Resolve cross-file dependencies in the repository, external API dependencies
   via predefined/dynamic API docs, and internal-file dependencies.
3. Format selected related information as:
   # {name} # {content}
"""

from __future__ import annotations
import os
from dataclasses import dataclass
from typing import List, Optional, Set

from referee.graph import ProjectContextGraph, CodeEntity
from referee.external_docs import ExternalDocRegistry, default_external_doc_registry


@dataclass
class RelatedInformationItem:
    """An item of related code information retrieved via dependency searching."""
    name: str
    content: str
    source_type: str  # 'internal', 'cross-file', 'external'
    entity_id: Optional[str] = None

    def format_paper_style(self) -> str:
        """
        Formats item in the style specified in Appendix A.2 (Step 6):
        '# {name} # {content}'
        """
        clean_content = self.content.strip()
        return f"# {self.name} #\n{clean_content}"


class CodeRelatedInfoSearcher:
    """
    Searches and extracts 1-hop code-related dependency information for an input code entity.
    """

    def __init__(
        self,
        graph: Optional[ProjectContextGraph] = None,
        external_docs: Optional[ExternalDocRegistry] = None,
    ):
        self.graph = graph or ProjectContextGraph()
        self.external_docs = external_docs or default_external_doc_registry

    def search_for_code(
        self,
        input_code: str,
        target_entity_name: Optional[str] = None,
        file_path: str = "main.py",
        project_root: Optional[str] = None,
    ) -> List[RelatedInformationItem]:
        """
        Performs 1-hop DFS on the project context graph starting from the input code
        to extract crucial related information (functions, classes, variables).
        """
        norm_file = os.path.normpath(file_path)
        # Add input code to graph if not already parsed
        entities = self.graph.add_python_source(input_code, file_path=norm_file)

        # Identify target entity
        target_entity: Optional[CodeEntity] = None
        if target_entity_name:
            target_entity = self.graph.find_entity_by_name(target_entity_name, norm_file)
        if not target_entity and entities:
            # Default to the primary function or class in the input snippet
            for ent in entities:
                if ent.ast_type in ("function_definition", "class_definition"):
                    target_entity = ent
                    break
            if not target_entity:
                target_entity = entities[0]

        if not target_entity:
            return []

        return self.search_for_entity(target_entity, project_root=project_root)

    def search_for_entity(
        self,
        target_entity: CodeEntity,
        project_root: Optional[str] = None,
    ) -> List[RelatedInformationItem]:
        """
        1-hop DFS starting from target_entity to find directly referenced entities.
        """
        results: List[RelatedInformationItem] = []
        seen_names: Set[str] = {target_entity.name}

        # 1-hop outgoing edges from target entity
        outgoing_edges = self.graph.outgoing.get(target_entity.id, [])

        for _, tail_id in outgoing_edges:
            tail_entity = self.graph.get_entity(tail_id)

            if tail_entity:
                # Case A: If tail entity is an import_statement
                if tail_entity.ast_type == "import_statement":
                    item = self._resolve_import(tail_entity, project_root)
                    if item and item.name not in seen_names:
                        seen_names.add(item.name)
                        results.append(item)

                # Case B: Internal-file dependency (function, class, assignment)
                elif tail_entity.file_path == target_entity.file_path:
                    if tail_entity.ast_type in ("function_definition", "class_definition", "assignment"):
                        if tail_entity.name not in seen_names:
                            content = self._extract_content(tail_entity)
                            if content:
                                seen_names.add(tail_entity.name)
                                results.append(
                                    RelatedInformationItem(
                                        name=tail_entity.name,
                                        content=content,
                                        source_type="internal",
                                        entity_id=tail_entity.id,
                                    )
                                )

                # Case C: Cross-file dependency already in graph
                else:
                    if tail_entity.ast_type in ("function_definition", "class_definition", "assignment"):
                        if tail_entity.name not in seen_names:
                            content = self._extract_content(tail_entity)
                            if content:
                                seen_names.add(tail_entity.name)
                                results.append(
                                    RelatedInformationItem(
                                        name=tail_entity.name,
                                        content=content,
                                        source_type="cross-file",
                                        entity_id=tail_entity.id,
                                    )
                                )

            else:
                # Unresolved tail_id (e.g. import_or_external:X or external:X)
                name = tail_id.split(":")[-1]
                if name not in seen_names:
                    # Check external docs
                    doc = self.external_docs.get_doc(name)
                    if doc:
                        seen_names.add(name)
                        results.append(
                            RelatedInformationItem(
                                name=name,
                                content=doc,
                                source_type="external",
                            )
                        )

        return results

    def _resolve_import(
        self,
        import_entity: CodeEntity,
        project_root: Optional[str] = None,
    ) -> Optional[RelatedInformationItem]:
        """
        Step (2)-(4):
        Transforms import into (module, name).
        Checks if module path exists in project directory -> cross-file dependency.
        Otherwise -> external dependency.
        """
        module = import_entity.metadata.get("module", "")
        symbol = import_entity.metadata.get("name", "")
        asname = import_entity.metadata.get("asname")
        display_name = asname or symbol or module

        # Check if cross-file in project directory
        root = project_root or self.graph.root_dir
        cross_file_match = self._find_cross_file_entity(module, symbol, root)
        if cross_file_match:
            content = self._extract_content(cross_file_match)
            if content:
                return RelatedInformationItem(
                    name=display_name,
                    content=content,
                    source_type="cross-file",
                    entity_id=cross_file_match.id,
                )

        # External dependency: check API docs
        doc = self.external_docs.get_doc(module, symbol)
        if doc:
            key_name = f"{module}.{symbol}" if (symbol and symbol != module) else module
            return RelatedInformationItem(
                name=key_name,
                content=doc,
                source_type="external",
            )

        return None

    def _find_cross_file_entity(
        self,
        module: str,
        symbol: str,
        project_root: Optional[str] = None,
    ) -> Optional[CodeEntity]:
        """Checks if module corresponds to a file in project and contains symbol."""
        # 1. Search in already indexed graph entities
        for ent in self.graph.entities.values():
            if ent.name == symbol and ent.ast_type in ("function_definition", "class_definition", "assignment"):
                # Check if module path matches file path
                mod_as_path = module.replace(".", os.sep)
                if mod_as_path in ent.file_path:
                    return ent

        # 2. Search on disk if project_root is provided
        if project_root and os.path.isdir(project_root):
            parts = module.split(".")
            candidate_files = [
                os.path.join(project_root, *parts) + ".py",
                os.path.join(project_root, *parts, "__init__.py"),
            ]
            for candidate in candidate_files:
                if os.path.isfile(candidate):
                    # Index and check
                    added = self.graph.add_python_file(candidate)
                    for ent in added:
                        if ent.name == symbol and ent.ast_type in (
                            "function_definition",
                            "class_definition",
                            "assignment",
                        ):
                            return ent

        return None

    def _extract_content(self, entity: CodeEntity) -> str:
        """
        As specified in Appendix A.2 (Step 6):
        'If the information corresponds to a class, the content contains the class docstring.
        For functions or variables, the content includes the code body.'
        """
        if entity.ast_type == "class_definition":
            if entity.docstring:
                return f"class {entity.name}(): {entity.docstring}"
            return entity.code_context
        return entity.code_context

    def format_related_information(self, items: List[RelatedInformationItem]) -> str:
        """
        Prepend format as in Appendix A.2 (Step 6):
        '# {name} # {content}'
        """
        if not items:
            return ""
        return "\n\n".join(item.format_paper_style() for item in items)

