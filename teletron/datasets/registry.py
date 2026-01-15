# Copyright (c) 2025 TeleAI-infra Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy

class Registry(dict):
    r"""A helper class for managing registering modules, it extends a
    dictionary and provides a register function."""

    def __init__(self, *args, **kwargs):
        super(Registry, self).__init__(*args, **kwargs)

    def register_module(self, module_class, module_name=None):
        if module_name is None:
            module_name = module_class.__name__
        assert module_name not in self, module_name
        self[module_name] = module_class
        return module_class

    def register(self, module_class):
        if isinstance(module_class, str):

            def _register(_module_class):
                return self.register_module(_module_class, module_class)

            return _register
        else:
            return self.register_module(module_class)


def merge_params(params_or_type, **kwargs):
    if isinstance(params_or_type, str):
        assert "type" not in kwargs
        kwargs["type"] = params_or_type
    elif isinstance(params_or_type, dict):
        params = copy.deepcopy(params_or_type)
        for name, value in params.items():
            assert name not in kwargs
            kwargs[name] = value
    else:
        assert params_or_type is None
    return kwargs


def build_module(registry, params_or_type, *args, **kwargs):
    params = merge_params(params_or_type, **kwargs)
    if "type" in params:
        obj_type = params.pop("type")
        assert obj_type in registry, "%s not in registry" % obj_type
        return registry[obj_type](*args, **params)
    else:
        return None
