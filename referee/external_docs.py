"""
Predefined and dynamic external API documentation provider for ReFEree.

As specified in Appendix A.2 (Step 4), when dependencies are imported from
external sources (e.g., Python built-ins, standard libraries, or third-party packages),
their descriptions are retrieved from predefined API documentation (such as official
Python docs, ODEX, DS-1000).
"""

from __future__ import annotations
import importlib
import inspect
import json
from typing import Any, Dict, List, Optional, cast


class ExternalDocRegistry:
    """Registry providing documentation for external libraries and APIs."""

    def __init__(self, predefined_docs: Optional[Dict[str, str]] = None):
        self.docs: Dict[str, str] = {}
        if predefined_docs:
            self.docs.update(predefined_docs)
        self._load_default_predefined()

    def register(self, key: str, docstring: str) -> None:
        """Register API documentation for an external module, class, or function."""
        self.docs[key] = docstring

    def load_from_json(self, file_path: str) -> None:
        """Load external documentation entries from a JSON file."""
        with open(file_path, "r", encoding="utf-8") as f:
            raw_data: Any = json.load(f)
            if isinstance(raw_data, dict):
                typed_dict: Dict[object, object] = cast(Dict[object, object], raw_data)
                for k, v in typed_dict.items():
                    if isinstance(k, str) and isinstance(v, str):
                        self.docs[k] = v

    def get_doc(self, module_name: str, symbol_name: Optional[str] = None) -> Optional[str]:
        """
        Retrieves documentation for (module_name, symbol_name) or (module_name.symbol_name).
        First checks predefined entries, then attempts dynamic inspection if available.
        """
        candidates: List[str] = []
        if symbol_name:
            candidates.append(f"{module_name}.{symbol_name}")
            candidates.append(symbol_name)
        candidates.append(module_name)

        # 1. Check registered predefined docs
        for cand in candidates:
            if cand in self.docs:
                return self.docs[cand]

        # 2. Dynamic inspection fallback using Python standard library / installed modules
        doc = self._try_inspect_doc(module_name, symbol_name)
        if doc:
            return doc

        return None

    def _try_inspect_doc(self, module_name: str, symbol_name: Optional[str] = None) -> Optional[str]:
        try:
            mod = importlib.import_module(module_name)
            obj: Any = mod
            if symbol_name and symbol_name != module_name:
                for part in symbol_name.split("."):
                    obj = getattr(obj, part)

            doc = inspect.getdoc(obj)
            if doc:
                sig_str = ""
                if callable(obj):
                    try:
                        sig = inspect.signature(obj)
                        target_name = symbol_name if (symbol_name and symbol_name != module_name) else module_name
                        sig_str = f"{target_name}{sig}: "
                    except Exception:
                        pass

                first_para = doc.strip().split("\n\n")[0].replace("\n", " ")
                return f"{sig_str}{first_para}"
        except Exception:
            pass
        return None

    def _load_default_predefined(self) -> None:
        """Pre-populate commonly used stdlib docs matching paper examples."""
        defaults = {
            "datetime.timedelta": (
                "class datetime.timedelta(): All arguments are optional and default to 0. "
                "A timedelta object represents a duration, the difference between two dates or times."
            ),
            "itertools.chain": (
                "itertools.chain(*iterables): Make an iterator that returns elements from the "
                "first iterable until it is exhausted, then proceeds to the next iterable, until "
                "all of the iterables are exhausted. This combines multiple data sources into a single iterator."
            ),
            "urllib.parse.urlsplit": (
                "urllib.parse.urlsplit(urlstring, scheme='', allow_fragments=True): "
                "Parse a URL into 5 components: <scheme>://<netloc>/<path>?<query>#<fragment>. "
                "Returns a 5-tuple: (scheme, netloc, path, query, fragment)."
            ),
            "urllib.parse.urlunsplit": (
                "urllib.parse.urlunsplit(parts): Combine the elements of a tuple as returned by "
                "urlsplit() into a complete URL as a string."
            ),
            "subprocess.Popen": (
                "subprocess.Popen(args, bufsize=-1, executable=None, stdin=None, stdout=None, stderr=None, ...): "
                "Execute a child program in a new process. On POSIX, the class uses os.execvpe()-like behavior."
            ),
            "inspect.getfullargspec": (
                "inspect.getfullargspec(func): Get the names and default values of a Python function's arguments. "
                "A tuple of 7 things is returned: (args, varargs, varkw, defaults, kwonlyargs, kwonlydefaults, annotations)."
            ),
            "sys.modules": (
                "sys.modules: A dictionary that maps module names to modules which have already been loaded."
            ),
            "math.log10": (
                "math.log10(x): Return the base 10 logarithm of x. This is usually more accurate than log(x, 10)."
            ),
        }
        for k, v in defaults.items():
            if k not in self.docs:
                self.docs[k] = v


default_external_doc_registry = ExternalDocRegistry()
