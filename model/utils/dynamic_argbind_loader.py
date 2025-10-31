# dynamic_argbind_loader.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import copy
import yaml
from typing import Any, Dict, List, Set, Union

Scalar = Union[str, int, float, bool, None]
Config = Dict[str, Any]

DEFAULT_INCLUDE_KEY = "$include"
DEFAULT_VAR_PREFIX = "&{"
DEFAULT_VAR_SUFFIX = "}"

class DynamicConfigLoader:
    """
    目标：
    - 支持 $include: [path1, path2, ...] 递归展开
    - 支持模板变量替换：&{var_name}
      * 替换来源优先级：overrides > 当前已解析的配置（同级/上级）
    - 安全措施：循环引用检测、相对路径基于父文件
    - 合并策略：
        dict: 深合并（后者覆盖前者）
        list: 默认覆盖（也可选择 append）
        scalar: 直接覆盖
    """
    def __init__(
        self,
        include_key: str = DEFAULT_INCLUDE_KEY,
        var_prefix: str = DEFAULT_VAR_PREFIX,
        var_suffix: str = DEFAULT_VAR_SUFFIX,
        list_strategy: str = "override",  # "override" | "append"
    ) -> None:
        assert list_strategy in ("override", "append")
        self.include_key = include_key
        self.var_prefix = var_prefix
        self.var_suffix = var_suffix
        self.list_strategy = list_strategy
        self.project_root = None

    # --------------------
    # 公共入口
    # --------------------
    def load_file(self, path: str, overrides: Config | None = None) -> Config:
        """
        从入口文件加载并解析动态 include 与变量替换。
        """
        path = os.path.abspath(path)
        self.project_root = os.path.dirname(os.path.dirname(path))
        overrides = copy.deepcopy(overrides or {})
        seen: Set[str] = set()
        cfg = self._load_recursive(path, parent_dir=os.path.dirname(path), seen=seen)
        # 变量替换（将 overrides 注入最高优先级）
        resolved = self._resolve_vars(cfg, overrides)
        # 再跑一轮 include（若变量替换后路径变得可解析），通常不需要；留空即可：
        return resolved

    # --------------------
    # 递归加载
    # --------------------
    def _load_recursive(self, path: str, parent_dir: str, seen: Set[str]) -> Config:
        if os.path.isabs(path):
            candidates = [path]
        else:
            # 候选1：相对当前文件目录
            cand1 = os.path.abspath(os.path.join(parent_dir, path))
            # 候选2：相对入口文件所在目录（工程根）
            cand2 = (os.path.abspath(os.path.join(self.project_root, path))
                     if self.project_root else cand1)
            candidates = [cand1] if cand1 == cand2 else [cand1, cand2]
            
        
        full = None
        for c in candidates:
            if os.path.exists(c):
                full = c
                break
        if full is None:
            tried = "\n  - ".join(candidates)
            raise FileNotFoundError(f"Config file not found for include '{path}'. Tried:\n  - {tried}")
        
        
        
        norm = os.path.normpath(full)

        if norm in seen:
            raise RuntimeError(f"Detected cyclic $include: {norm}")
        if not os.path.exists(norm):
            raise FileNotFoundError(f"Config file not found: {norm}")

        seen.add(norm)
        with open(norm, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        if not isinstance(raw, dict):
            raise TypeError(f"Top-level YAML must be a mapping, got: {type(raw)} in {norm}")

        # 先取出 includes
        includes = raw.pop(self.include_key, [])
        if isinstance(includes, (str,)):
            includes = [includes]
        if includes and not isinstance(includes, list):
            raise TypeError(f"{self.include_key} must be a list or string in {norm}")

        base_dir = os.path.dirname(norm)
        merged: Config = {}

        # 先合并上游 include
        for inc in includes:
            if not isinstance(inc, str):
                raise TypeError(f"Each item of {self.include_key} must be a string in {norm}")
            # 注意：此时 inc 可能包含模板变量，先做一次“就地变量替换（仅来自 raw）”
            inc_resolved = self._replace_in_string(inc, raw)
            child = self._load_recursive(inc_resolved, parent_dir=base_dir, seen=seen)
            merged = self._merge(merged, child)

        # 再合并当前文件内容
        merged = self._merge(merged, raw)

        return merged

    # --------------------
    # 变量替换（深度）
    # --------------------
    def _resolve_vars(self, cfg: Any, overrides: Config) -> Any:
        """
        深度变量替换：在字符串中替换 &{key}。
        替换源：overrides 优先，其次是 cfg 自身（使用“点路径”查值）。
        """
        def _resolve_node(node: Any, root: Config) -> Any:
            if isinstance(node, dict):
                return {k: _resolve_node(v, root) for k, v in node.items()}
            elif isinstance(node, list):
                return [_resolve_node(v, root) for v in node]
            elif isinstance(node, str):
                return self._replace_in_string(node, root, overrides)
            else:
                return node

        # 为了能在解析时通过“点路径”查找，做一个可查函数闭包
        self._root_snapshot = cfg  # 保留一份解析前的 root
        return _resolve_node(cfg, cfg)

    def _replace_in_string(self, s: str, local_scope: Config, overrides: Config | None = None) -> str:
        """
        将 s 中的 &{var} 替换为对应值（转字符串）。
        覆盖优先级：overrides > local_scope > self._root_snapshot
        支持点路径 a.b.c。
        """
        if self.var_prefix not in s:
            return s

        result = ""
        i = 0
        while i < len(s):
            start = s.find(self.var_prefix, i)
            if start == -1:
                result += s[i:]
                break
            # copy plain part
            result += s[i:start]
            end = s.find(self.var_suffix, start)
            if end == -1:
                # 无闭合，按原样返回剩余
                result += s[start:]
                break
            key = s[start + len(self.var_prefix): end].strip()
            val = self._lookup_with_priority(key, local_scope, overrides)
            result += str(val)
            i = end + len(self.var_suffix)
        return result

    def _lookup_with_priority(self, key: str, local_scope: Config, overrides: Config | None) -> Any:
        # 优先：overrides
        if overrides:
            found, v = self._lookup_dotpath(overrides, key)
            if found:
                return v
        # 其次：local_scope（当前合并层）
        found, v = self._lookup_dotpath(local_scope, key)
        if found:
            return v
        # 最后：root 快照（全局）
        root = getattr(self, "_root_snapshot", None)
        if root is not None:
            found, v = self._lookup_dotpath(root, key)
            if found:
                return v
        # 如果没找到，保持原样（返回占位符本身，避免硬崩）
        return f"{self.var_prefix}{key}{self.var_suffix}"

    @staticmethod
    def _lookup_dotpath(obj: Any, path: str) -> (bool, Any):
        """
        在嵌套 dict/list 中用 a.b.c 点路径取值。
        支持 a.0.b 访问列表项。
        """
        cur = obj
        for part in path.split("."):
            if isinstance(cur, dict):
                if part in cur:
                    cur = cur[part]
                else:
                    return False, None
            elif isinstance(cur, list):
                if part.isdigit():
                    idx = int(part)
                    if 0 <= idx < len(cur):
                        cur = cur[idx]
                    else:
                        return False, None
                else:
                    return False, None
            else:
                return False, None
        return True, cur

    # --------------------
    # 合并策略
    # --------------------
    def _merge(self, base: Any, incoming: Any) -> Any:
        if isinstance(base, dict) and isinstance(incoming, dict):
            out = dict(base)
            for k, v in incoming.items():
                if k in out:
                    out[k] = self._merge(out[k], v)
                else:
                    out[k] = copy.deepcopy(v)
            return out
        elif isinstance(base, list) and isinstance(incoming, list):
            if self.list_strategy == "append":
                return copy.deepcopy(base) + copy.deepcopy(incoming)
            else:
                # override
                return copy.deepcopy(incoming)
        else:
            # scalar or type mismatch -> incoming wins
            return copy.deepcopy(incoming)


# --------------------
# 对外的便捷函数
# --------------------
def load_config_for_argbind(
    main_yaml: str,
    overrides: Config | None = None,
    list_strategy: str = "override",
    include_key: str = DEFAULT_INCLUDE_KEY,
) -> Config:
    """
    一步到位：加载 + 解析 + 返回 dict，用于 argbind.parse_args(..., config=cfg)
    """
    loader = DynamicConfigLoader(
        include_key=include_key,
        list_strategy=list_strategy,
    )
    return loader.load_file(main_yaml, overrides=overrides or {})
