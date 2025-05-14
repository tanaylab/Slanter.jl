push!(LOAD_PATH, ".")

using Aqua
using Slanter
Aqua.test_ambiguities([Slanter])
Aqua.test_all(Slanter; ambiguities = false, unbound_args = false, deps_compat = false, persistent_tasks = false)
